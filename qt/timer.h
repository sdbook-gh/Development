#pragma once

#include <QWidget>
#include <QProgressBar>

class Timer : public QWidget {
    Q_OBJECT
public:
    Timer(QWidget *parent = 0);
public slots:
    void updateProgress();
private:
    QProgressBar *progressBar {nullptr};
};
