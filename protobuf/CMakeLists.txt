# temp_dir=$(mktemp -d) temp_proto=$temp_dir/amessage.proto && cat c.proto > $temp_proto && cat amessage.proto >> $temp_proto && ./protobuf-c-1.5.0/install/bin/protoc-c $temp_proto -I$temp_dir --c_out=.
# temp_dir=$(mktemp -d) temp_proto=$temp_dir/amessage.proto && cat cpp.proto > $temp_proto && cat amessage.proto >> $temp_proto && ./protobuf-3.20.1/install/bin/protoc amessage.proto -I$temp_dir --cpp_out=.

set(CMAKE_CXX_STANDARD 17)
find_package(Threads REQUIRED)
add_executable(test_pc main.cpp amessage.pb-c.c amessage.pb.cc)
target_include_directories(test_pc PRIVATE protobuf-c-1.5.0/install/include protobuf-3.20.1/install/include)
target_link_directories(test_pc PRIVATE protobuf-c-1.5.0/install/lib protobuf-3.20.1/install/lib)
target_link_libraries(test_pc PRIVATE protobuf-c protobuf)
