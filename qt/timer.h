#pragma once

#include <QWidget>
#include <QProgressBar>
#include <QTimer>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>

class Timer : public QWidget {
    Q_OBJECT
public:
    Timer(QWidget *parent = 0);
public slots:
    void updateProgress();
private:
    //QProgressBar *progressBar {nullptr};
    QLabel *_qc_label {nullptr};
    QComboBox *_qc_type {nullptr};
    QLabel *_lt_lable {nullptr};
    QComboBox *_lt_type {nullptr};
    QLabel *_ls_lable {nullptr};
    QLineEdit *_ls_content {nullptr};
    QLabel *_li_lable {nullptr};
    QLineEdit *_li_content {nullptr};
    QPushButton *_filter_button {nullptr};
};
