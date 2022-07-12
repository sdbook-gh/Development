#include <QApplication>
#include <iostream>

#include "mainwindow.h"
#include "test_signal_slot.h"

#include "timer.h"

int main(int argc, char *argv[]) {
	QApplication app(argc,argv);

	/*MainWindow mainWindow;
	mainWindow.resize(600, 600);
	mainWindow.show();*/

	Timer timer;
    timer.show();

	std::cout << "in func main" << std::endl;
	return app.exec();
}
