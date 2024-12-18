#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include "mywidget.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT

  public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
  private slots:
    void slotTimeOut();

  private:
    Ui::MainWindow *ui;
    MyGLWidget *_mywidget;
    QTimer timer;
};

#endif // MAINWINDOW_H
