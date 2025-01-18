cmake_minimum_required(VERSION 3.0)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 COMPONENTS Core)
find_package(Qt6 COMPONENTS Gui)
find_package(Qt6 COMPONENTS Concurrent)
find_package(Qt6 COMPONENTS Widgets)

qt_add_executable(wordcount
    main.cpp
)
target_link_libraries(wordcount PUBLIC
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Widgets
)
