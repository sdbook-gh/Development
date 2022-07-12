#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <sstream>

MainWindow::MainWindow(QWidget* parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow),
	signal_tester(&slot_tester) {
	ui->setupUi(this);
	ui->centralwidget->setMouseTracking(true);
	ui->label->setVisible(false);

	/*
	startPoint = new QPoint(100, 100);
	startRect = new QRect(QPoint(95, 95), QPoint(105, 105));

	endPoint = new QPoint(200, 200);
	endRect = new QRect(QPoint(195, 195), QPoint(205, 205));

	c1 = new QPoint(180, 50);
	c1Rect = new QRect(QPoint(175, 45), QPoint(185, 55));

	c2 = new QPoint(120, 250);
	c2Rect = new QRect(QPoint(115, 245), QPoint(125, 255));

	this->setMouseTracking(true);
	*/
	_timer = new QTimer(this);
	_timer->setTimerType(Qt::PreciseTimer);//设置定时器的精度，精确到毫秒
	connect(_timer, SIGNAL(timeout()), this, SLOT(slotTimeOut()));
	_timer->start(5000);
}

#include <cmath>
#define CURVE_POINT_SIZE (3)
#define KNOT_NUM (3)
#define PI std::acos(-1.0f)
struct CurvePoint {
	double x;
	double y;
};
double cal_angle(CurvePoint point1, CurvePoint point2) {
	double angle_temp;
	double xx, yy;
	xx = point2.x - point1.x;
	yy = point2.y - point1.y;
	if (xx == 0.0)
		angle_temp = PI / 2.0;
	else
		angle_temp = std::atan(std::fabs(yy / xx));
	if ((xx < 0.0) && (yy >= 0.0))
		angle_temp = PI - angle_temp;
	else if ((xx < 0.0) && (yy < 0.0))
		angle_temp = PI + angle_temp;
	else if ((xx >= 0.0) && (yy < 0.0))
		angle_temp = PI * 2.0 - angle_temp;
	return (angle_temp * 180.0 / PI);
}
CurvePoint rotate(const CurvePoint& heart, CurvePoint& point, double angle) {
	double xx = (point.x - heart.x) * std::cos(angle * PI / 180) -
		(point.y - heart.y) * std::sin(angle * PI / 180) + heart.x;
	double yy = (point.y - heart.y) * std::cos(angle * PI / 180) +
		(point.x - heart.x) * std::sin(angle * PI / 180) + heart.y;
	point.x = xx;
	point.y = yy;
	return point;
}
void cal_cicular(CurvePoint curve_points[CURVE_POINT_SIZE], CurvePoint& heart, float& R, std::vector<CurvePoint>& out_points, std::vector<CurvePoint>& out_points2, std::vector<CurvePoint>& out_points_rectangle) {
	CurvePoint tmpCurvePoint = curve_points[0];
	curve_points[0] = curve_points[2];
	curve_points[2] = tmpCurvePoint;
	double x1, y1, x2, y2, x3, y3;
	double a, b, c, g, e, f;
	double X, Y;
	x1 = curve_points[0].x;
	y1 = curve_points[0].y;
	x2 = curve_points[1].x;
	y2 = curve_points[1].y;
	x3 = curve_points[2].x;
	y3 = curve_points[2].y;
	e = 2 * (x2 - x1);
	f = 2 * (y2 - y1);
	g = x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1;
	a = 2 * (x3 - x2);
	b = 2 * (y3 - y2);
	c = x3 * x3 - x2 * x2 + y3 * y3 - y2 * y2;
	X = (g * b - c * f) / (e * b - a * f);
	Y = (a * g - c * e) / (a * f - b * e);
	heart.x = X;
	heart.y = Y;
	printf("heart: %f %f\n", X, Y);
	R = std::sqrt((X - x1) * (X - x1) + (Y - y1) * (Y - y1));

	double angoa = cal_angle(heart, curve_points[0]);
	//angoa = angoa > 180.0 ? angoa - 180.0 : angoa;
	printf("+++++++ angoa: %f\n", angoa);
	double angob = cal_angle(heart, curve_points[1]);
	//angob = angob > 180.0 ? angob - 180.0 : angob;
	printf("+++++++ angob: %f\n", angob);
	double angoc = cal_angle(heart, curve_points[2]);
	//angoc = angoc > 180.0 ? angoc - 180.0 : angoc;
	printf("+++++++ angoc: %f\n", angoc);
	std::vector<double> vec_radians;
	for (int i = 0; i <= KNOT_NUM; ++i) {
		double div_angel = angoa - angob;
		if (div_angel < 0) {
			div_angel += 360.0;
		}
		double angel = angoa - div_angel / (KNOT_NUM + 1) * i;
		if (angel < 0.0) {
			angel += 360.0;
		}
		vec_radians.push_back(angel);
	}
	for (int i = 0; i <= KNOT_NUM; ++i) {
		double div_angel = angob - angoc;
		if (div_angel < 0) {
			div_angel += 360.0;
		}
		double angel = angob - div_angel / (KNOT_NUM + 1) * i;
		if (angel < 0.0) {
			angel += 360.0;
		}
		vec_radians.push_back(angel);
	}
	for (int i = 0; i < vec_radians.size(); ++i) {
		printf("+++++++ vec_radians: %f\n", vec_radians[i]);
		//x1 = x0 + r * cos(ao * 3.14 / 180)
		//y1 = y0 + r * sin(ao * 3.14 / 180)
		double x = X + R * std::cos(vec_radians[i] * (PI / 180.0));
		double y = Y + R * std::sin(vec_radians[i] * (PI / 180.0));
		out_points.push_back(CurvePoint{ x, y });
		out_points2.push_back(CurvePoint{ x, y });
	}
	//out_points.push_back(curve_points[2]);
	//out_points2.push_back(curve_points[2]);

	/*double angle = cal_angle(curve_points[0], curve_points[2]);
	printf("+++++++ angle: %f\n", angle);
	out_points.push_back(
		rotate(curve_points[2], CurvePoint{ curve_points[2].x, curve_points[2].y - 50 }, angle));
	out_points.push_back(
		rotate(curve_points[0], CurvePoint{ curve_points[0].x, curve_points[0].y - 50 }, angle));
	out_points.push_back(curve_points[0]);*/
}

void MainWindow::paintEvent(QPaintEvent* event) {
	QFont time_font;
	time_font.setPointSize(40);
	QPainter time_painter(this);
	time_painter.setFont(time_font);
	time_painter.setPen(QPen(Qt::green, 2));
	//time_painter.drawText(this->width() / 2, this->height() / 2, _time_string);
	time_painter.drawText(0, 0, this->width(), this->height(), Qt::AlignCenter, _time_string);
	return;
	/*printf("angel -- : %f\n", cal_angle(CurvePoint{4,4}, CurvePoint{80, 80}));

	printf("angel: %f\n", cal_angle(CurvePoint{ 0,0 }, CurvePoint{ 80, 20 }));
	printf("angel: %f\n", cal_angle(CurvePoint{ 0,0 }, CurvePoint{ 200, 30 }));
	printf("angel: %f\n", cal_angle(CurvePoint{ 0,0 }, CurvePoint{ 40, 200 }));*/

	QPainter painter(this);
	painter.setWindow(-300, 300, 600, -600);
	painter.setPen(QPen(Qt::green, 2));
	painter.drawPoint(0, 0);

	//CurvePoint points[3]{ {20, 100}, {50, 50}, {80, 100} };
	//CurvePoint points[3]{ {200, 30}, {80, 20}, {40, 200} };
	//CurvePoint points[3]{ {40, 200} , {80, 20}, {200, 30} };
	//CurvePoint points[3]{ {20, 100} , {50, 50}, {80, 30} };
	//CurvePoint points[3]{ {80, 30} , {50, 50}, {20, 100} };
	//CurvePoint points[3]{ {220, 30} , {250, 50}, {280, 100} };
	//CurvePoint points[3]{ {280, 100}  , {250, 50}, {220, 30} };

	//CurvePoint points[3]{ {30, 66} , {53, 84}, {78, 67} };
	//CurvePoint points[3]{ {33, 35} , {42, 30}, {34, 22} };
	//CurvePoint points[3]{ {73, 12} , {67, 05}, {62, 12} };
	//CurvePoint points[3]{ {76, 86} , {72, 93}, {77, 100} };
	//CurvePoint points[3]{ {71, 29} , {78, 38}, {91, 29} };
	//CurvePoint points[3]{ {02, 98} , {21, 74}, {07, 47} };

	CurvePoint points[3]{ {131, 151} , {127, 188}, {81, 199} };
	for (int i = 0; i < sizeof(points) / sizeof(points[0]); i++) {
		painter.setPen(QPen(Qt::red, 2 + 2 * i));
		painter.drawPoint(points[i].x, points[i].y);
	}
	CurvePoint heart;
	float R;
	std::vector<CurvePoint> out_points;
	std::vector<CurvePoint> out_points2;
	std::vector<CurvePoint> out_rectangle;
	cal_cicular(points, heart, R, out_points, out_points2, out_rectangle);
	painter.setPen(QPen(Qt::black, 1));
	//painter.drawEllipse(QPointF(heart.x, heart.y), R, R);
	//for (int i = 0; i < out_points2.size(); i++) {
		//painter.setPen(QPen(Qt::blue, 2));
		//painter.drawPoint(out_points2[i].x, out_points2[i].y);
		//printf("-- %f %f --\n", out_points2[i].x, out_points2[i].y);
	//}
	for (int i = 0; i < out_points.size(); i++) {
		painter.setPen(QPen(Qt::green, 2 + i));
		painter.drawPoint(out_points[i].x, out_points[i].y);
		//printf("-- %f %f --\n", out_points[i].x, out_points[i].y);
	}
	painter.setPen(QPen(Qt::black, 1));
	QVector<QLineF> lines;
	int size = out_points.size();
	for (int i = 0; i < size - 1; i++) {
		QLineF line(QPointF(out_points[i].x, out_points[i].y), QPointF(out_points[i + 1].x, out_points[i + 1].y));
		lines.push_back(line);
	}
	//QLineF line(QPointF(out_points[size - 1].x, out_points[size - 1].y), QPointF(out_points[0].x, out_points[0].y));
	//lines.push_back(line);
	painter.drawLines(lines);

	/*static std::vector<CurvePoint> out_points;
	static std::vector<CurvePoint> control_points;

	out_points.clear();
	createCurve(points, out_points, control_points);

	painter.setPen(QPen(Qt::black, 1));
	printf("out_points size:%d\n", out_points.size());
	for (int i = 0; i < out_points.size() - 1; i++) {
		painter.drawLine(out_points[i].x, out_points[i].y, out_points[i + 1].x, out_points[i + 1].y);
	}
	painter.setPen(QPen(Qt::green, 4));
	for (int i = 0; i < control_points.size(); i++) {
		painter.drawPoint(control_points[i].x, control_points[i].y);
	}*/

	/*out_points.clear();
	createCurveNew(points, out_points);

	painter.setPen(QPen(Qt::black, 1));
	printf("out_points size:%d\n", out_points.size());
	for (int i = 0; i < out_points.size() - 1; i++) {
		painter.drawLine(out_points[i].x, out_points[i].y, out_points[i + 1].x, out_points[i + 1].y);
	}*/

	/*
	QPainter painter(this);
	painter.setRenderHint(QPainter::Antialiasing, true);
	QPen pen;
	pen.setColor(QColor(255, 0, 0));
	pen.setWidth(_penWidth);
	pen.setStyle(Qt::SolidLine);
	painter.setPen(pen);

	drawBeser(painter);
	*/
}

void MainWindow::mousePressEvent(QMouseEvent* event) {
	/*
	if (event->button() == Qt::LeftButton) {
		_leftDown = true;

	}
	update();
	*/
}

void MainWindow::mouseMoveEvent(QMouseEvent* event) {
	/*
	QPoint pos = event->pos();
	if (_leftDown) {//左键按下
		if (_selected) {//并且选中点
			_selectPoint->setX(pos.x());
			_selectPoint->setY(pos.y());
		}
	}
	else {
		if (startRect->contains(pos)) {
			_penWidth = 4;
			_selected = true;
			_selectPoint = startPoint;
		}
		else if (endRect->contains(pos)) {
			_penWidth = 4;
			_selected = true;
			_selectPoint = endPoint;
		}
		else if (c1Rect->contains(pos)) {
			_penWidth = 4;
			_selected = true;
			_selectPoint = c1;
		}
		else if (c2Rect->contains(pos)) {
			_penWidth = 4;
			_selected = true;
			_selectPoint = c2;
		}
		else {
			_selected = false;
			_penWidth = 2;
			_selectPoint = nullptr;
		}
	}

	update();
	*/
}

void MainWindow::mouseReleaseEvent(QMouseEvent* event) {
	/*
	if (event->button() == Qt::LeftButton) {
		_leftDown = false;
	}
	else if (event->button() == Qt::RightButton) {
	}
	update();
	*/
}

void MainWindow::drawBeser(QPainter& painter) {
	QPainterPath pathTest;
	painter.drawEllipse(*startPoint, 2, 2);
	painter.drawEllipse(*endPoint, 2, 2);
	painter.drawEllipse(*c1, 2, 2);
	painter.drawEllipse(*c2, 2, 2);
	pathTest.moveTo(*startPoint);
	pathTest.cubicTo(*c1, *c2, *endPoint);

	pointPercent = pathTest.pointAtPercent(_percent);
	painter.drawEllipse(pointPercent, 5, 5);

	painter.drawPath(pathTest);

}

void MainWindow::slotTimeOut() {
	//_percent += 0.01;
	time_t tnow = time(0);
	std::tm* ptm = localtime(&tnow); // gmtime for UTC
	std::stringstream ss;
	ss << std::put_time(ptm, "%Y-%m-%d %H:%M:%S");
	_time_string = ss.str().c_str();
	update();
	signal_tester.do_test();
}

MainWindow::~MainWindow() {
	delete ui;
}
