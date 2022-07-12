#include <QLayout>

#include "timer.h"

Timer::Timer(QWidget *parent)
    : QWidget(parent) {
    auto *layout = new QHBoxLayout();
    /*progressBar = new QProgressBar();
    progressBar->setMinimum(0);
    progressBar->setMaximum(100);
    _ls_lable = new QLabel(tr("Source"));
    _ls_content = new QLineEdit;
    //_ls_content->setMaximumWidth(50);
    _ls_content->setText("");
    _li_lable = new QLabel(tr("Info"));
    _li_content = new QLineEdit;
    //_li_content->setMaximumWidth(50);
    _li_content->setText("");
    _filter_button = new QPushButton(tr("确定"));
    layout->addWidget(progressBar);
    layout->addWidget(_ls_lable);
    layout->addWidget(_ls_content);
    layout->addWidget(_li_lable);
    layout->addWidget(_li_content);
    layout->addWidget(_filter_button);
    QTimer *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &Timer::updateProgress);
    timer->start(1000);*/

    auto *qc_widget = new QWidget(this);
    qc_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    qc_widget->setMaximumHeight(20);
    auto *qc_layout = new QHBoxLayout(qc_widget);
    _qc_label = new QLabel(tr("强制质检"));
    _qc_type = new QComboBox;
    QStringList ql;
    ql.push_back(tr("是"));
    ql.push_back(tr("否"));
    ql.push_back(tr("全部"));
    ql.clear();
    _qc_type->addItems(ql);
    _qc_type->setCurrentIndex(2);
    _lt_lable = new QLabel(tr("Type"));
    _lt_type = new QComboBox;
    ql.push_back(tr("INFO"));
    ql.push_back(tr("WARNING"));
    ql.push_back(tr("ERROR"));
    ql.push_back(tr("全部"));
    ql.clear();
    _lt_type->addItems(ql);
    _lt_type->setCurrentIndex(3);
    _ls_lable = new QLabel(tr("Source"));
    _ls_content = new QLineEdit;
    //_ls_content->setMaximumWidth(50);
    _ls_content->setText("");
    _li_lable = new QLabel(tr("Info"));
    _li_content = new QLineEdit;
    //_li_content->setMaximumWidth(50);
    _li_content->setText("");
    _filter_button = new QPushButton(tr("确定"));
    qc_layout->addWidget(_qc_label);
    qc_layout->addWidget(_qc_type);
    qc_layout->addWidget(_lt_lable);
    qc_layout->addWidget(_lt_type);
    qc_layout->addWidget(_ls_lable);
    qc_layout->addWidget(_ls_content);
    qc_layout->addWidget(_li_lable);
    qc_layout->addWidget(_li_content);
    qc_layout->addWidget(_filter_button);
    qc_layout->setAlignment(Qt::AlignCenter);
    qc_layout->setMargin(0);
    qc_layout->setContentsMargins(0, 0, 0, 0);
    qc_widget->setLayout(qc_layout);
    layout->addWidget(qc_widget);
    setLayout(layout);
    setWindowTitle(tr("Timer and progress"));
    //resize(10, 10);
}

void Timer::updateProgress() {
    //progressBar->setValue(progressBar->value() + 1);
}
