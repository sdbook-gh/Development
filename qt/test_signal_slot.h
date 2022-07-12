#pragma once
#include <QObject>

/*#include <QObject>
#include <QDialog>
#include <QPushButton>
#include <QTextEdit>
#include <QTimer>

class Example : public QObject
{
    Q_OBJECT
public:
    explicit Example(QObject* parent = nullptr);

signals:
    void test(const QString& text);

public slots:
    void onButtonClicked(bool checked);
    void onTimeout();

    void onTest1(const QString& text);
    void onTest2(const QString& text);
    void onOpenInvite();

private:
    void setUp();
    void events();

    void addText(const QString& text);

    QString getTime();

    QDialog m_dialog;
    QPushButton* m_button;
    QTextEdit* m_textEdit;
    QTimer m_timer;
};*/

class TestSlot;

class TestSignal : public QObject {
	Q_OBJECT
signals:
	void do_signal();
public:
	explicit TestSignal(TestSlot* pslot, QObject* parent = nullptr);
	void do_test();
};

class TestSlot : public QObject {
    Q_OBJECT
public slots:
    void do_slot();
public:
    explicit TestSlot(QObject* parent = nullptr);
};
