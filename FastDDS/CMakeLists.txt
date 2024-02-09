
set(CMAKE_CXX_STANDARD 17)

add_executable(fastdds_pub HelloWorldPublisher.cpp idl_out/HelloWorldPubSubTypes.cxx idl_out/HelloWorld.cxx)
target_include_directories(fastdds_pub PUBLIC idl_out /home/shenda/NETA/code/Fast-DDS_install/include)
target_link_directories(fastdds_pub PUBLIC /home/shenda/NETA/code/Fast-DDS_install/lib /home/shenda/NETA/code/foonathan_memory_vendor_install/lib)
target_link_libraries(fastdds_pub fastcdr fastrtps foonathan_memory)

add_executable(fastdds_sub HelloWorldSubscriber.cpp idl_out/HelloWorldPubSubTypes.cxx idl_out/HelloWorld.cxx)
target_include_directories(fastdds_sub PUBLIC idl_out /home/shenda/NETA/code/Fast-DDS_install/include)
target_link_directories(fastdds_sub PUBLIC /home/shenda/NETA/code/Fast-DDS_install/lib /home/shenda/NETA/code/foonathan_memory_vendor_install/lib)
target_link_libraries(fastdds_sub fastcdr fastrtps foonathan_memory)
