project(helloqt)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui)

qt5_wrap_ui(uifiles mainwindow.ui)
qt5_add_resources(rcfiles ${CMAKE_CURRENT_SOURCE_DIR}/qrc/demo.qrc)

set(project_sources demomain.cpp timer.cpp mainwindow.cpp test_signal_slot.cpp)
add_executable(helloqt ${project_sources} ${rcfiles} ${uifiles})

target_link_libraries(helloqt PUBLIC
	Qt5::Core
	Qt5::Gui
	Qt5::Widgets
)
