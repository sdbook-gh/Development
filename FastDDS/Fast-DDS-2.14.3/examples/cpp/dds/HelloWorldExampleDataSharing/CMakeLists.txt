# Copyright 2019 Proyectos y Sistemas de Mantenimiento SL (eProsima).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)

project(DDSHelloWorldExampleDataSharing VERSION 1 LANGUAGES CXX)

# Find requirements
if(NOT fastcdr_FOUND)
    find_package(fastcdr REQUIRED)
endif()

if(NOT fastrtps_FOUND)
    find_package(fastrtps 2.12 REQUIRED)
endif()

#Check C++11
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag(-std=c++11 SUPPORTS_CXX11)
    if(NOT SUPPORTS_CXX11)
        message(FATAL_ERROR "Compiler doesn't support C++11")
    endif()
endif()

message(STATUS "Configuring HelloWorld example...")
file(GLOB DDS_HELLOWORLD_EXAMPLE_DATASHARING_SOURCES_CXX "*.cxx")
file(GLOB DDS_HELLOWORLD_EXAMPLE_DATASHARING_SOURCES_CPP "*.cpp")

add_executable(DDSHelloWorldExampleDataSharing
    ${DDS_HELLOWORLD_EXAMPLE_DATASHARING_SOURCES_CXX}
    ${DDS_HELLOWORLD_EXAMPLE_DATASHARING_SOURCES_CPP})
target_link_libraries(DDSHelloWorldExampleDataSharing fastrtps fastcdr)
install(TARGETS DDSHelloWorldExampleDataSharing
    RUNTIME DESTINATION examples/cpp/dds/HelloWorldExampleDataSharing/${BIN_INSTALL_DIR})
