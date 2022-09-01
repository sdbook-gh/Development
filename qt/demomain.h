#pragma once

#include <QStyledItemDelegate>
 
class MyDelegate : public QStyledItemDelegate {
    Q_OBJECT
public:
    explicit MyDelegate(QWidget *parent = 0);
 
    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const;
	void setStyleOption(QStyleOptionViewItem &option, const QModelIndex &index, const QStyleOptionViewItem &old_option) const;
    // QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const;
    // QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
    // void setEditorData(QWidget *editor, const QModelIndex &index) const;
    // void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
};

#include <QMetaType>
#include <QRect>
#include <QPainter>
#include <QImage>

class MyPic {
public:
    MyPic();
    void paint(QPainter *painter,const QRect &rect) const;
};
 
Q_DECLARE_METATYPE(MyPic)

#include <QVariant>
#include <QStandardItemModel>
class MyModel : public QStandardItemModel
{
public:
    MyModel();
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;
};
