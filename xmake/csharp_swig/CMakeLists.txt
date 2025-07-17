cmake_minimum_required(VERSION 4.0)
add_library(animal SHARED animal.cpp swig/animalCSHARP_wrap.cxx)
target_include_directories(animal PRIVATE ${CMAKE_CURRENT_LIST_DIR})
install(TARGETS animal
    RUNTIME DESTINATION ${CMAKE_CURRENT_LIST_DIR}
    LIBRARY DESTINATION ${CMAKE_CURRENT_LIST_DIR}
)
