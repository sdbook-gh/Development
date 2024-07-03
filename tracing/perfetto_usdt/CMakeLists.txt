set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(sdk)
add_library(perfetto STATIC sdk/perfetto.cc)
find_package(protobuf REQUIRED CONFIG)
add_executable(test_pb test_pb.cpp amessage.pb.cc)
target_link_libraries(test_pb protobuf::libprotobuf perfetto)
