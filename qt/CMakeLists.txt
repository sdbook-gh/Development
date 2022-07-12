#set PROJECT_NAME and version
project(HelloQT)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui)

set(project_ui mainwindow.ui)
set(project_headers mainwindow.h)
set(project_sources demomain.cpp mainwindow.cpp test_signal_slot.cpp)

#set(project_timer_src timer.cpp demomain.cpp)

add_executable(helloqt ${project_headers} ${project_ui} ${project_sources})

#add_executable(helloqt ${project_timer_src})

# Use the widgets module from Qt 5
target_link_libraries(helloqt PUBLIC
	Qt5::Core
	Qt5::Gui
	Qt5::Widgets
)
