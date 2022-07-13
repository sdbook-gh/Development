#include <QApplication>
#include <iostream>
#include <QTableWidget>
#include <QHeaderView>
#include <QStringList>
#include <qDebug>

#include "mainwindow.h"
#include "test_signal_slot.h"
#include "timer.h"
#include "demomain.h"
 
MyDelegate::MyDelegate(QWidget *parent) : QStyledItemDelegate(parent) {
}
void MyDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const {
    if(index.data().canConvert<MyPic>()) {
        MyPic mypic=qvariant_cast<MyPic>(index.data());
        mypic.paint(painter,option.rect);
    } else {
        qDebug()<<"can't be convert!";
    }
    QStyledItemDelegate::paint(painter, option, index);
}
QSize MyDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const {
    return QStyledItemDelegate::sizeHint(option,index);
}
QWidget *MyDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const {
    return nullptr;
}
void MyDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const {
}
void MyDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const {
    QStyledItemDelegate::setModelData(editor,model,index);
}
MyPic::MyPic() {
}
void MyPic::paint(QPainter *painter,const QRect &rect) const {
    painter->drawImage(rect.x(),rect.y(),QImage(":/icon/dog.png").scaled(30,30));
}

int main(int argc, char *argv[]) {
	QApplication app(argc,argv);

	MainWindow mainWindow;
	mainWindow.resize(600, 600);
	mainWindow.show();

	/*Timer timer;
    timer.show();*/

	/*QTableWidget tableWidget(4,5);
    QStringList headerList;
    headerList<<"header1"<<"header2"<<"header3"<<"header4"<<"header5";
    tableWidget.setHorizontalHeaderLabels(headerList);
    //it make beautiful header
    for(int i=0;i<headerList.size();i++)
        tableWidget.horizontalHeader()->setSectionResizeMode(i,QHeaderView::Stretch);
    for(int i=0;i<tableWidget.rowCount();i++){
        for(int j=0;j<tableWidget.columnCount();j++){
            QTableWidgetItem *item=new QTableWidgetItem;
            item->setData(0,QVariant::fromValue(MyPic()));
            tableWidget.setItem(i,j,item);
        }
    }
    tableWidget.setItemDelegate(new MyDelegate);
    tableWidget.resize(600,150);
    tableWidget.show();*/

	std::cout << "in func main" << std::endl;
	return app.exec();
}
