"""Acquire, independently audit, retry incomplete native labels, then verify frozen recommendations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import fcntl
from pathlib import Path
import time

from .mission_audit import audit_closed
from .mission_campaign import OUT, WORK, read, register, run_one
from .mission_release import build


def progress(last_count):
    completed=sum(1 for _ in (OUT/'acquisition').glob('*/attempt_*.json'))
    if completed==last_count:
        return completed
    audit_closed()
    if completed:
        build('progress')
        audits=[read(p) for p in (OUT/'audits').glob('*/attempt_*.json')]
        print(f'{sum(a["accepted"] for a in audits)} verified flights from {completed} closed attempts',flush=True)
    return completed


def acquire(planned,path,workers):
    for number in (1,2,3):
        queue=[]
        for c in planned:
            cid=c['case_id']
            if (OUT/'acquisition'/cid/f'attempt_{number}.json').exists():
                continue
            if number>1:
                previous=OUT/'audits'/cid/f'attempt_{number-1}.json'
                if not previous.exists() or read(previous)['accepted']:
                    continue
            queue.append(c)
        if not queue:
            audit_closed()
            continue
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(run_one,c,path,number) for c in queue}
            count=-1
            while futures:
                done,futures=wait(futures,timeout=25,return_when='FIRST_COMPLETED')
                for completed in done:
                    completed.result()
                count=progress(count)
        audit_closed()


def external_acquisition(pid):
    cmd=Path('/proc')/str(pid)/'cmdline'
    try:
        args=cmd.read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    return b'tools.m100_reaudit.mission_campaign' in args and b'--confirmation' not in args


def run(workers,attach):
    if not 1<=workers<=4:
        raise ValueError('Choose one to four workers for this host')
    if (OUT/'verification.json').exists():
        audit_closed()
        build('final')
        return
    parent=register()
    count=-1
    while attach and external_acquisition(attach):
        count=progress(count)
        time.sleep(25)
    acquire(parent['cases'],OUT/'protocol.json',workers)
    if not (OUT/'confirmation_protocol.json').exists():
        build('freeze')
    confirmation=read(OUT/'confirmation_protocol.json')
    acquire(confirmation['cases'],OUT/'confirmation_protocol.json',workers)
    build('final')
    print('Verified release',read(OUT/'verification.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--attach-discovery-pid',type=int)
    args=p.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.workers,args.attach_discovery_pid)
