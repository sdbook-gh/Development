#pragma once

#include <QMainWindow>
#include <QPainter>
#include <QPoint>
#include <QRect>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QTimer>
#include <QPainterPath>
#include "test_signal_slot.h"

namespace Ui {
	class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT
public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow();

	void paintEvent(QPaintEvent* event)override;
	void mousePressEvent(QMouseEvent* event)override;
	void mouseMoveEvent(QMouseEvent* event)override;
	void mouseReleaseEvent(QMouseEvent* event)override;

private slots:
	void slotTimeOut();

private:
	Ui::MainWindow* ui;

	void drawBeser(QPainter& painter);

	bool _leftDown = false;//鼠标左键按下

	int _penWidth = 1;

	QPoint* startPoint = nullptr;
	QRect* startRect = nullptr;

	QPoint* endPoint = nullptr;
	QRect* endRect = nullptr;

	QPoint* c1 = nullptr;
	QRect* c1Rect = nullptr;

	QPoint* c2 = nullptr;
	QRect* c2Rect = nullptr;

	QPainterPath _pathTest;

	bool _selected = false;//是否选中点
	QPoint* _selectPoint = nullptr;

	QPointF pointPercent;//百分之比的点

	QTimer* _timer = nullptr;
	double _percent = 0.0;
	QString _time_string;
	TestSlot slot_tester;
	TestSignal signal_tester;

};
