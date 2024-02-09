# find_package(Protobuf REQUIRED)
# include_directories(${Protobuf_INCLUDE_DIRS})
# include_directories(${CMAKE_CURRENT_BINARY_DIR})
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_LIST_DIR}/examples.proto)
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS EXPORT_MACRO DLL_EXPORT foo.proto)
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS DESCRIPTORS PROTO_DESCS foo.proto)

include_directories(${CMAKE_CURRENT_LIST_DIR} /home/shenda/CyberRT /home/shenda/CyberRT/install/include)
link_directories(/home/shenda/CyberRT/build /home/shenda/CyberRT/install/lib /home/shenda/CyberRT/build)

set(LINK_LIBS cyber fastrtps glog protobuf gflags pthread)

add_executable(channel_test_writer channel_test_writer.cc examples.pb.cc)
target_link_libraries(channel_test_writer
    ${LINK_LIBS}
)

add_executable(channel_prediction_writer channel_prediction_writer.cc examples.pb.cc)
target_link_libraries(channel_prediction_writer
    ${LINK_LIBS}
)

add_library(common_component_example SHARED common_component_example.cc examples.pb.cc)
target_link_libraries(common_component_example
  ${LINK_LIBS}
)
set_target_properties(common_component_example PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(COPY common.dag common.launch
     DESTINATION ${CMAKE_BINARY_DIR}
)
