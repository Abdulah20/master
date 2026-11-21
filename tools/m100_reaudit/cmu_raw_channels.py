"""Summarise recorded channels in CMU development and ancillary ROS1 bags, offline."""

import argparse
import csv
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/abdulah/Master/Dataset /DJI Matrice 100')
REFERENCE_MD5 = '351de11d5e7db5521606a931ea10ce63'
BATTERY_MD5 = '476f837fa6771f6e16e3bf4ef96f8770'
BATTERY_FIELDS = (
    'Header header', 'float32 voltage', 'float32 current', 'float32 charge',
    'float32 capacity', 'float32 design_capacity', 'float32 percentage',
    'uint8 power_supply_status', 'uint8 power_supply_health',
    'uint8 power_supply_technology', 'bool present', 'float32[] cell_voltage',
    'string location', 'string serial_number',
)
NAMES = [field.split()[1] for field in BATTERY_FIELDS]
UINT32 = struct.Struct('<I')
SCALARS = struct.Struct('<6f3B?')


def digest(path, algorithm):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def block(buf, offset):
    size, = UINT32.unpack_from(buf, offset)
    start = offset + UINT32.size
    if start + size > len(buf):
        raise ValueError('Truncated length-prefixed ROS1 block')
    return buf[start:start + size], start + size


def header_fields(buf):
    header = {}
    offset = 0
    while offset < len(buf):
        field, offset = block(buf, offset)
        name, value = bytes(field).split(b'=', 1)
        header[name] = value
    return header


def record_at(buf, offset):
    header, offset = block(buf, offset)
    payload, offset = block(buf, offset)
    return header_fields(header), payload, offset


def records(buf, offset=0):
    while offset < len(buf):
        header, payload, offset = record_at(buf, offset)
        yield header, payload


def battery_connection(meta):
    main = meta[b'message_definition'].decode().split('====', 1)[0]
    lines = [line.split('#', 1)[0].strip() for line in main.splitlines()]
    fields = tuple(' '.join(line.split()) for line in lines if line and '=' not in line)
    if fields != BATTERY_FIELDS or meta[b'md5sum'].decode() != BATTERY_MD5:
        raise ValueError('Unsupported recorded BatteryState schema')


def value_key(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+inf' if value > 0 else '-inf')
    return value


def battery_values(buf):
    # ROS1 Header is seq, stamp.sec, stamp.nsec, then a length-prefixed frame_id.
    _, offset = block(buf, 12)
    values = SCALARS.unpack_from(buf, offset)
    offset += SCALARS.size
    cells, = UINT32.unpack_from(buf, offset)
    offset += UINT32.size
    cell_voltages = struct.unpack_from(f'<{cells}f', buf, offset)
    location, offset = block(buf, offset + cells * 4)
    serial, offset = block(buf, offset)
    if offset != len(buf):
        raise ValueError('Unexpected bytes at end of BatteryState')
    return (*values, tuple(map(value_key, cell_voltages)), bytes(location).decode(), bytes(serial).decode())


def numeric_summary(hist):
    items = sorted((v, n) for v, n in hist.items() if isinstance(v, float))
    total = sum(n for _, n in items)
    positions = ((total - 1) // 2, total // 2)
    cumulative = 0
    middle = []
    for v, n in items:
        middle.extend(v for p in positions if cumulative <= p < cumulative + n)
        cumulative += n
    return {
        'finite': total, 'nonfinite_counts': {v: n for v, n in hist.items() if isinstance(v, str)},
        'min': items[0][0] if total else None, 'median': sum(middle) / 2 if total else None,
        'max': items[-1][0] if total else None, 'negative': sum(n for v, n in items if v < 0),
        'zero': hist[0.0], 'positive': sum(n for v, n in items if v > 0),
    }


def bag_channels(data, signatures, auxiliary):
    magic = b'#ROSBAG V2.0\n'
    if not data.startswith(magic):
        raise ValueError('Expected ROS1 bag format version 2.0')
    buf = memoryview(data)
    file_header, _, _ = record_at(buf, len(magic))
    index_pos, = struct.unpack('<Q', file_header[b'index_pos'])
    outer = list(records(buf, index_pos))
    connections = {}
    chunks = {}
    starts = []
    for header, payload in outer:
        if header[b'op'] == b'\x06':  # CHUNK_INFO: one position and a per-connection message count table
            pos, = struct.unpack('<Q', header[b'chunk_pos'])
            starts.append(struct.unpack('<II', header[b'start_time']))
            chunks[pos] = dict(struct.iter_unpack('<II', payload))
            if len(chunks[pos]) != UINT32.unpack(header[b'count'])[0]:
                raise ValueError('Incomplete chunk information table')
        if header[b'op'] == b'\x07':  # CONNECTION records at the end of an indexed bag
            meta = header_fields(payload)
            topic = header[b'topic'].decode()
            connections[UINT32.unpack(header[b'conn'])[0]] = topic
            signature = (meta[b'type'].decode(), meta[b'md5sum'].decode(), meta.get(b'callerid', b'').decode())
            signatures[topic].add(signature)
            if topic == '/battery':
                if signature[0] != 'sensor_msgs/BatteryState':
                    raise ValueError(f'Unexpected battery connection type: {signature[0]}')
                battery_connection(meta)

    if len(chunks) != UINT32.unpack(file_header[b'chunk_count'])[0] or len(connections) != UINT32.unpack(file_header[b'conn_count'])[0]:
        raise ValueError('Chunk or connection count differs from the bag header')
    voltage, current, topic_counts = Counter(), Counter(), Counter({topic: 0 for topic in connections.values()})
    positions = sorted(chunks)
    padding = 0
    for pos, limit in zip(positions, [*positions[1:], index_pos], strict=True):
        header, chunk, end = record_at(buf, pos)
        if header[b'op'] != b'\x05' or header[b'compression'] != b'none' or len(chunk) != UINT32.unpack(header[b'size'])[0]:
            raise ValueError('Expected an uncompressed ROS chunk of its declared size')
        observed = Counter()
        for record, message in records(chunk):
            if record[b'op'] != b'\x02':  # MSG_DATA
                continue
            conn, = UINT32.unpack(record[b'conn'])
            observed[conn] += 1
            topic = connections[conn]
            topic_counts[topic] += 1
            if topic == '/battery':
                v, i, *other = battery_values(message)
                voltage[value_key(v)] += 1
                current[value_key(i)] += 1
                for field, val in zip(NAMES[3:], other, strict=True):
                    auxiliary[field][value_key(val)] += 1
        indices = {}
        for _ in chunks[pos]:
            header, payload, end = record_at(buf, end)
            conn, = UINT32.unpack(header[b'conn'])
            indices[conn], = UINT32.unpack(header[b'count'])
            if header[b'op'] != b'\x04' or len(payload) != 12 * indices[conn]:
                raise ValueError('Invalid ROS1 per-connection time/offset index')
        if dict(observed) != chunks[pos] or dict(observed) != indices:
            raise ValueError('Recorded messages and bag-index counts differ')
        if end > limit or any(buf[end:limit]):
            raise ValueError('Unexplained bytes between indexed chunks or end index')
        padding += limit - end
    secs, nanos = min(starts)
    return topic_counts, voltage, current, padding, datetime.fromtimestamp(secs + nanos / 1e9, timezone.utc)


def audit(archive_path, parameters, fit):
    with parameters.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    routes = {int(row['flight']): row['route'] for row in rows}
    dates = {int(row['flight']): row['date'] for row in rows}
    cohorts = json.loads(fit.read_text())['cohorts']
    groups = {key: cohorts[key]['flight_ids'] for key in ('train', 'validation')}
    groups['ancillary_ground'] = list(range(211, 220))
    authorized = [i for ids in groups.values() for i in ids]
    if len(set(authorized)) != len(authorized) or any(routes[i] not in ('A1', 'A2', 'A3') for i in groups['ancillary_ground']):
        raise ValueError('Overlapping cohorts or unexpected ancillary metadata')

    raw_md5 = digest(archive_path, 'md5')  # Hash the compressed ZIP bytes without decoding its members.
    if raw_md5 != REFERENCE_MD5:
        raise ValueError('Archive differs from the pinned published artifact')
    signatures, auxiliary = defaultdict(set), defaultdict(Counter)
    padding_by_id = {}
    group_results, ground_records, clock_examples = {}, {}, {}
    with ZipFile(archive_path) as archive:
        names = Counter(entry.filename for entry in archive.infolist())
        bag_ids = [int(m[1]) for name in names if (m := re.fullmatch(r'raw/(\d+)/raw\.bag', name))]
        if len(bag_ids) != len(set(bag_ids)) or any(n != 1 for n in names.values()):
            raise ValueError('Duplicated member name or bag ID')
        available = set(bag_ids)
        for group, ids in groups.items():
            totals, present = Counter(), Counter()
            voltage, current = Counter(), Counter()
            varying, utc_days, site_days = Counter(), Counter(), Counter()
            matching_dates = 0
            for i in sorted(set(ids) & available):
                count, v, a, pad, start = bag_channels(archive.read(f'raw/{i}/raw.bag'), signatures, auxiliary)
                if pad:
                    padding_by_id[str(i)] = pad
                local_date = start.astimezone(ZoneInfo('America/New_York')).date().isoformat()
                utc_days[start.date().isoformat()] += 1
                site_days[local_date] += 1
                matching_dates += local_date == dates[i]
                if i in (1, 178, 211):
                    clock_examples[str(i)] = {'first_indexed_record_UTC': start.isoformat(), 'publisher_local_date': dates[i]}
                present.update(count.keys())
                totals.update(count)
                voltage.update(v)
                current.update(a)
                varying['voltage'] += sum(isinstance(x, float) for x in v) > 1
                varying['current'] += sum(isinstance(x, float) for x in a) > 1
                if group == 'ancillary_ground':
                    ground_records[str(i)] = {'route': routes[i], 'battery_messages': count['/battery'],
                                              'voltage_V': numeric_summary(v), 'current_A': numeric_summary(a)}
            group_results[group] = {
                'opened_ids': sorted(set(ids) & available), 'requested_ids_without_bag': sorted(set(ids) - available),
                'bags_with_connection': dict(present), 'topic_messages': dict(totals),
                'bags_with_multiple_finite_values': dict(varying), 'voltage_V': numeric_summary(voltage),
                'current_A_unmodified': numeric_summary(current),
                'bag_clock_start_dates_UTC': dict(utc_days), 'bag_clock_start_dates_America_New_York': dict(site_days),
                'bag_clock_local_date_matching_published_date': matching_dates,
            }
        inventory = {
            'zip_bytes': archive_path.stat().st_size, 'zip_md5': raw_md5, 'matches_published_md5': True,
            'zip_entries': sum(names.values()), 'bag_members': len(bag_ids), 'duplicate_member_names': [],
            'duplicate_bag_ids': [], 'metadata_rows': len(rows), 'metadata_distinct_ids': len(routes),
            'metadata_ids_without_bag': sorted(routes.keys() - available), 'bag_ids_without_metadata': sorted(available - routes.keys()),
            'bags_opened': sum(len(g['opened_ids']) for g in group_results.values()),
            'other_bags_not_opened': len(available - set(authorized)),
            'zero_padding_after_chunk_indices_bytes_by_id': padding_by_id,
        }
    return {
        'source': 'Rodrigues et al., CMU, https://doi.org/10.1184/R1/12683453',
        'method': 'ZIP directory covers all members. Only named development and ancillary bags are opened, one in memory at a time. Walk the chunks referenced by the bag header/end index. For each chunk, match the count of complete messages per connection to both its time/offset index and CHUNK_INFO table. Extra bytes between indexed areas are required to be zero padding. Decode only /battery. Summaries pool all messages including time on the ground. Median and sign counts use native signed samples, without resampling.',
        'inputs': {'archive_path': str(archive_path), 'parameters_path': str(parameters), 'parameters_sha256': digest(parameters, 'sha256'),
                   'cohorts_path': str(fit), 'cohorts_sha256': digest(fit, 'sha256'), 'reference_md5_source': 'https://api.figshare.com/v2/articles/12683453, version 1, raw_files.zip file ID 26405936'},
        'inventory': inventory, 'groups': group_results, 'ancillary_records': ground_records,
        'bag_clock_examples': clock_examples,
        'all_opened_connection_signatures': {
            topic: [{'message_type': s[0], 'md5': s[1], 'publisher': s[2] or None} for s in sorted(variants)]
            for topic, variants in sorted(signatures.items())},
        'battery_definition': {'md5': BATTERY_MD5, 'top_level_fields_in_wire_order': list(BATTERY_FIELDS),
                               'absent_from_this_schema': ['temperature', 'cell_temperature']},
        'other_battery_fields_all_opened_messages': {
            k: [{'value': v, 'messages': n} for v, n in sorted(hist.items(), key=lambda item: repr(item[0]))]
            for k, hist in auxiliary.items()},
        'interpretation': {
            'source_instrument': 'The publisher identifies the Mauch PL-200 and ADC and documents only voltage and current as populated (README.txt lines 134, 147–152). The schema specifies negative current on discharge. The sign counts here describe the bytes as recorded.',
            'observed_field_values': 'In the inspected records, large currents are positive. Charge, capacity, design_capacity and percentage carry literal zeros on every message. Voltage/current vary despite present=false throughout. Unknown status/health/technology and empty cell/identity fields supply no independent SOC, capacity, pack identity or cell labels.',
            'motion': '/nav/odom represents aircraft odometry, /imu/data aircraft inertial motion, and /anemometer local airflow. None is a per-rotor speed, ESC input/PWM, shaft torque or battery pulse command.',
            'ground': 'The authors describe A1 as no propellers or motor movement, A2 as no propellers with minimum motor movement and A3 as propellers with minimum motor movement. These are system-level ancillary cases. No controlled battery-pulse/rest protocol or measured shaft load is specified (README.txt lines 434–438).',
            'dates': 'Bag start times use the earliest indexed record, with the publisher site time zone used for date comparisons. Use published flight-ID metadata for calendar splits; the absolute bag clock needs an independently established time reference.',
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DATA / 'raw_files.zip')
    parser.add_argument('--parameters', type=Path, default=ROOT / 'data/external/m100/parameters.csv')
    parser.add_argument('--cohorts', type=Path, default=ROOT / 'results/m100_reaudit/cmu_calibration/fit.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/m100_reaudit/cmu_component_evidence/raw_channels.json')
    args = parser.parse_args()
    result = audit(args.archive, args.parameters, args.cohorts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
