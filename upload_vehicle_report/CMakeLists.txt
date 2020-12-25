cmake_minimum_required(VERSION 3.5)
project(upload_vehicle_report CXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# find dependencies
find_package(rclcpp REQUIRED)
include_directories(${rclcpp_INCLUDE_DIRS})

find_package(lifecycle_msgs REQUIRED)
include_directories(${lifecycle_msgs_INCLUDE_DIRS})

find_package(ados_vehicle_msgs REQUIRED)
include_directories(${ados_vehicle_msgs_INCLUDE_DIRS})

find_package(Threads REQUIRED)

find_package(yaml-cpp REQUIRED)
include_directories(${yaml-cpp_INCLUDE_DIRS})
find_library(yaml-cpp_STATIC_LIBRARIES libyaml-cpp.a)

find_package(PkgConfig REQUIRED)
pkg_check_modules(openssl REQUIRED openssl)
include_directories(${openssl_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src DIR_SRC)
add_executable(uploader ${DIR_SRC})
target_link_libraries(uploader ${rclcpp_LIBRARIES} ${lifecycle_msgs_LIBRARIES} ${ados_vehicle_msgs_LIBRARIES} Threads::Threads ${yaml-cpp_STATIC_LIBRARIES} ${openssl_LIBRARIES} ${boost-container})

install(TARGETS uploader DESTINATION lib/upload_vehicle_report/)
