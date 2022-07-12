#include "test_signal_slot.h"
#include <thread>
#include <sstream>

/*#include <QDateTime>
#include <QDebug>
#include <QLabel>
#include <QString>

Example::Example(QObject* parent) :
    QObject(parent)
{
    setUp();

    events();
}

void Example::onButtonClicked(bool checked)
{
    Q_UNUSED(checked)

        m_button->setEnabled(false);
    emit test("This is an sitnal test");
}

void Example::onTimeout()
{
    addText(" Timer has triggered");
}

void Example::onTest1(const QString& text)
{
    addText(text + " written out by onTest1");
    QTimer::singleShot(2000, this, &Example::onOpenInvite);
}

void Example::onTest2(const QString& text)
{
    addText(text + " written out by onTest2");
}

void Example::onOpenInvite()
{
    QDialog dialog;

    addText("open new dialog.");

    dialog.setWindowTitle("Invitation");
    QRect geometry = m_dialog.geometry();
    dialog.setGeometry(QRect(geometry.topRight(), QSize(595, 842)));

    QLabel* label = new QLabel(&dialog);
    label->setGeometry(QRect(0, 0, 595, 842));
    QPixmap pixmap = QPixmap(":resources/images/invitation.png");
    label->setPixmap(pixmap);

    int counter = 0;
    QTimer timer;
    timer.setInterval(1000);
    connect(&timer, &QTimer::timeout,
        this,
        [&]() {
            dialog.setWindowTitle(
                QString("Invitation. timer: %1").arg(counter++));
        });

    timer.start();
    dialog.exec();

    m_button->setEnabled(true);
}

void Example::setUp()
{
    m_timer.setInterval(2000);
    m_timer.start();

    m_dialog.setWindowTitle("Test Dialog");
    m_dialog.setMinimumSize(QSize(640, 480));

    m_button = new QPushButton(&m_dialog);
    m_button->setText("Start");
    m_button->setGeometry(QRect(270, 450, 100, 30));

    m_textEdit = new QTextEdit(&m_dialog);
    m_textEdit->setTextInteractionFlags(Qt::NoTextInteraction);
    m_textEdit->setGeometry(QRect(0, 0, 640, 450));

    onTimeout();

    m_dialog.show();
}

void Example::events()
{
    connect(m_button, &QPushButton::clicked, this, &Example::onButtonClicked);
    connect(this, &Example::test, this, &Example::onTest1);
    connect(this, &Example::test, this, &Example::onTest2);
    connect(m_button, &QPushButton::clicked,
        this, [&]() { m_dialog.setWindowTitle(m_dialog.windowTitle() + "+"); });
    connect(&m_timer, &QTimer::timeout, this, &Example::onTimeout);
}

void Example::addText(const QString& text)
{
    m_textEdit->append(getTime() + " " + text);
}

QString Example::getTime()
{
    return QDateTime::currentDateTime().time().toString("HH:mm:ss");
}*/

TestSignal::TestSignal(TestSlot* pslot, QObject* parent) : QObject(parent) {
	connect(this, &TestSignal::do_signal, pslot, &TestSlot::do_slot);
}
void TestSignal::do_test() {
    std::stringstream ss;
    ss << std::this_thread::get_id();
    printf("do_test %s\n", ss.str().c_str());
	emit do_signal();
}

TestSlot::TestSlot(QObject* parent) : QObject(parent) {
}
void TestSlot::do_slot() {
    std::stringstream ss;
    ss << std::this_thread::get_id();
    printf("do_slot %s\n", ss.str().c_str());
}
