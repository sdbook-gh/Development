#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      _mywidget(0) {
    ui->setupUi(this);
    _mywidget = new MyGLWidget(this);
    ui->gridLayout->addWidget(_mywidget, 0, 0, 1, 1);
    _mywidget->stackUnder(ui->widget);
    _mywidget->show();
    connect(&timer, SIGNAL(timeout()), this, SLOT(slotTimeOut()));
    timer.setTimerType(Qt::PreciseTimer);
    timer.start(100);
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::slotTimeOut() {
    QImage noImage;
    if (!noImage.load(tr("/apollo/modules/tools/visualizer/images/no_image.png"))) {
        std::cout << "--------can not load the default texture------------\n";
        return;
    }
    _mywidget->setImage(noImage);
    _mywidget->repaint(); // 窗口重绘，repaint会调用paintEvent函数，paintEvent会调用paintGL函数实现重绘
}
