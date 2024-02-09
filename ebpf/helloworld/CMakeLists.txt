include(ExternalProject)

execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory bpftool
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                TIMEOUT 5
                RESULT_VARIABLE result_var
                OUTPUT_VARIABLE output_var
                ERROR_VARIABLE error_var
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE
)

ExternalProject_Add(bpftool
                    PREFIX bpftool
                    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libbpf-bootstrap/bpftool/src
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND make bootstrap
                    OUTPUT=${CMAKE_CURRENT_SOURCE_DIR}/bpftool/
                    BUILD_IN_SOURCE TRUE
                    INSTALL_COMMAND ""
                    STEP_TARGETS build
)

ExternalProject_Add(helloworld
                    PREFIX helloworld
                    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND make
                    OUTPUT=""
                    BUILD_IN_SOURCE TRUE
                    INSTALL_COMMAND ""
                    STEP_TARGETS build
)
