cmake_minimum_required(VERSION 3.22)

project(uav_energy_gazebo_plugin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(gz-sim8 8.11.0 EXACT REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

add_library(MotorBatteryLoadSystem SHARED
  src/MotorBatteryLoadSystem.cpp
)

target_include_directories(MotorBatteryLoadSystem
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(MotorBatteryLoadSystem
  PRIVATE
    gz-sim8::gz-sim8
    gz-plugin2::register
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10
)

target_compile_options(MotorBatteryLoadSystem PRIVATE -Wall -Wextra -Wpedantic -Werror)

include(CTest)

if(BUILD_TESTING)
  add_executable(power_model_test tests/power_model_test.cpp)
  target_include_directories(power_model_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(power_model_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME power_model_test COMMAND power_model_test)
endif()
