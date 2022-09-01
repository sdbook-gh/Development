#include <QApplication>
#include <QHeaderView>
#include <QObject>
#include <QPalette>
#include <QStringList>
#include <QStyle>
#include <QTableWidget>
#include <QTextDocument>
#include <iostream>
#include <qDebug>

#include "demomain.h"
#include "mainwindow.h"
#include "test_signal_slot.h"
#include "timer.h"

static const QColor force_bg(200, 200, 200);
static const QColor focus_bg(100, 100, 100);

MyDelegate::MyDelegate(QWidget *parent) : QStyledItemDelegate(parent) {}
void MyDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option,
                       const QModelIndex &index) const {
  if (index.data().canConvert<MyPic>()) {
    QStyledItemDelegate::paint(painter, option, index);
    MyPic mypic = qvariant_cast<MyPic>(index.data());
    mypic.paint(painter, option.rect);
  } else if (index.data().canConvert<QString>()) {
    painter->save();
    auto content = qvariant_cast<QString>(index.data(0));
    QStyleOptionViewItem viewOption(option);
    // setStyleOption(viewOption, index, option);
    // viewOption.state &= ~QStyle::State_HasFocus;
    bool focused = option.state.testFlag(QStyle::State_HasFocus);
    bool selected = option.state.testFlag(QStyle::State_Selected);
    const QWidget *widget = option.widget;
    // printf("widget: %x\n", widget);
    QStyle *style = widget ? widget->style() : QApplication::style();
    if (content == "force") {
      if (!focused) {
        viewOption.text = content;
        style->drawControl(QStyle::CE_ItemViewItem, &viewOption, painter,
                           widget);
        QPen pen;
        pen.setColor(Qt::green);
        pen.setWidth(5);
        painter->setPen(pen);
        painter->drawRect(option.rect);
      } else {
        QTextDocument doc;
        QString richtext_content = "<font color=\"#FF00FF\">";
        richtext_content += content;
        richtext_content += "<\font>";
        doc.setHtml(richtext_content);
        painter->translate(viewOption.rect.left(), viewOption.rect.top());
        QRect clip(0, 0, viewOption.rect.width(), viewOption.rect.height());
        doc.drawContents(painter, clip);
        QPen pen;
        pen.setColor(focus_bg);
        pen.setWidth(5);
        painter->setPen(pen);
        painter->drawRect(option.rect);
      }
    } else {
      if (!focused) {
        viewOption.text = content;
        style->drawControl(QStyle::CE_ItemViewItem, &viewOption, painter,
                           widget);
      } else if (selected) {
        viewOption.text = content;
        viewOption.displayAlignment = Qt::AlignCenter | Qt::AlignVCenter;
        viewOption.backgroundBrush = Qt::green;
        QPalette &pt = viewOption.palette;
        pt.setBrush(QPalette::Text, Qt::black);
        pt.setBrush(QPalette::HighlightedText, Qt::red);
        pt.setBrush(QPalette::Base, Qt::cyan);
        pt.setBrush(QPalette::Highlight, Qt::darkCyan);
        style->drawControl(QStyle::CE_ItemViewItem, &viewOption, painter,
                           nullptr);
      } else {
        viewOption.text = content;
        viewOption.displayAlignment = Qt::AlignRight | Qt::AlignVCenter;
        viewOption.backgroundBrush = Qt::red;
        QPalette &pt = viewOption.palette;
        pt.setBrush(QPalette::Text, Qt::white);
        pt.setBrush(QPalette::HighlightedText, Qt::yellow);
        pt.setBrush(QPalette::Base, Qt::cyan);
        pt.setBrush(QPalette::Highlight, Qt::darkCyan);
        // QApplication::style()->drawItemText(painter, viewOption.rect,
        // viewOption.displayAlignment, viewOption.palette, true, content,
        // QPalette::Text);
        // QApplication::style()->drawPrimitive(QStyle::PE_Widget, &viewOption,
        // painter, widget);
        style->drawControl(QStyle::CE_ItemViewItem, &viewOption, painter,
                           widget);
      }
    }
    painter->restore();
  } else {
    qDebug() << "can't be convert!";
  }
}
void MyDelegate::setStyleOption(QStyleOptionViewItem &option,
                                const QModelIndex &index,
                                const QStyleOptionViewItem &old_option) const {
  //字体
  // QVariant value = index.data(Qt::FontRole);
  // if (value.isValid() && !value.isNull()) {
  // option.font = qvariant_cast<QFont>(value).resolve(option.font);
  // option.fontMetrics = QFontMetrics(option.font);
  // }
  //对齐方式
  // value = index.data(Qt::TextAlignmentRole);
  // if (value.isValid() && !value.isNull()) {
  // option.displayAlignment = Qt::Alignment(value.toInt());
  // }
  //也可以直接全部指定为居中对齐
  // option.displayAlignment = Qt::AlignCenter;
  //前景色
  // value = index.data(Qt::ForegroundRole);
  // if (value.canConvert<QBrush>())
  // option.palette.setBrush(QPalette::Text, qvariant_cast<QBrush>(value));
  // option.index = index;
  // value = index.data(Qt::CheckStateRole); 未使用，暂略
  // value = index.data(Qt::DecorationRole); 未使用，暂略
  //文本
  // value = index.data(Qt::DisplayRole);
  // if (value.isValid() && !value.isNull()) {
  // option.features |= QStyleOptionViewItem::HasDisplay;
  // }
  // if (index.data(0).canConvert<QString>()) {
  // option.text = qvariant_cast<QString>(index.data(0));
  // }
  //背景色
  // option.backgroundBrush =
  // qvariant_cast<QBrush>(index.data(Qt::BackgroundRole));
  // option.backgroundBrush = QBrush(force_bg);
  // disable style animations for checkboxes etc. within itemviews (QTBUG-30146)
  // QWidget *srcWidget = qobject_cast<QWidget *>(old_option.styleObject);
  // QStyle *style = srcWidget ? srcWidget->style() : QApplication::style();
  // option.styleObject = style;
  // option.styleObject = old_option.styleObject;
}
// QSize MyDelegate::sizeHint(const QStyleOptionViewItem &option,
// const QModelIndex &index) const {
// return QStyledItemDelegate::sizeHint(option, index);
// }
// QWidget *MyDelegate::createEditor(QWidget *parent,
// const QStyleOptionViewItem &option,
// const QModelIndex &index) const {
// return nullptr;
// }
// void MyDelegate::setEditorData(QWidget *editor,
// const QModelIndex &index) const {}
// void MyDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
// const QModelIndex &index) const {
// QStyledItemDelegate::setModelData(editor, model, index);
// }

MyPic::MyPic() {}
void MyPic::paint(QPainter *painter, const QRect &rect) const {
  painter->drawImage(rect.x(), rect.y(),
                     QImage(":/icon/dog.png").scaled(30, 30));
}

MyModel::MyModel() {}

QVariant MyModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid())
    return QVariant();
  if (role == Qt::TextAlignmentRole) {
    return int(Qt::AlignLeft | Qt::AlignVCenter);
  } else if (role == Qt::DisplayRole) {
    return QStandardItemModel::data(index, role);
  } else if (role == Qt::BackgroundRole) {
    if (index.row() % 5 == 0)
      return QColor(Qt::red);
    else if (index.row() % 5 == 1)
      return QColor(Qt::green);
    else if (index.row() % 5 == 2)
      return QColor(Qt::blue);
    else if (index.row() % 5 == 3)
      return QColor(Qt::yellow);
    else if (index.row() % 5 == 4)
      return QColor(Qt::gray);
    else
      return QVariant();
  }
  return QVariant();
}

int main(int argc, char *argv[]) {
  qDebug(QT_VERSION_STR);
  QApplication app(argc, argv);
  QFile qss(":/qss/skin.qss");
  auto suc = qss.open(QFile::ReadOnly);
  QTextStream filetext(&qss);
  QString stylesheet = filetext.readAll();
  app.setStyleSheet(stylesheet);

  MyModel *model = new MyModel();
  model->setColumnCount(2);
  model->setHeaderData(0, Qt::Horizontal, QObject::tr("货品名称"));
  model->setHeaderData(1, Qt::Horizontal, QObject::tr("货品类型"));
  QTableView otherTable;
  otherTable.setStyleSheet("QTableView {"
                           "  background: blue;"
                           "}"
                           "QTableView::item {"
                           "  background: green;"
                           "}");
  otherTable.setModel(model);
  otherTable.setColumnWidth(0, 100);
  otherTable.setColumnWidth(1, 100);
  for (int i = 0; i < 10; i++) {
    if (i % 2 != 0) {
      model->setItem(i, 0, new QStandardItem("ddd"));
      model->setItem(i, 1, new QStandardItem("哈哈"));
      // model->data(model->index(i,1),Qt::BackgroundRole);//使用此方法也能改变背景色，但是只能改变某个item无法改变一整行？
    } else {
      model->setItem(i, 0, new QStandardItem("ddd"));
      model->setItem(i, 1, new QStandardItem("哈哈"));
    }
  }
  otherTable.resize(600, 500);
  otherTable.show();

  // MainWindow mainWindow;
  // mainWindow.resize(600, 600);
  // mainWindow.show();

  // Timer timer;
  // timer.show();

  QTableWidget tableWidget(10, 5);
  // tableWidget.setObjectName("Good");
  // tableWidget.setStyleSheet(
  // "QTableView {"
  // "  background: blue;"
  // "}"
  // "QTableView::item {"
  // "  background: green;"
  // "}"
  // );
  QStringList headerList;
  headerList << "header1"
             << "header2"
             << "header3"
             << "header4"
             << "header5";
  tableWidget.setHorizontalHeaderLabels(headerList);
  // it make beautiful header
  for (int i = 0; i < headerList.size(); i++) {
    tableWidget.horizontalHeader()->setSectionResizeMode(i,
                                                         QHeaderView::Stretch);
  }
  for (int i = 0; i < tableWidget.rowCount(); i++) {
    for (int j = 0; j < tableWidget.columnCount(); j++) {
      QTableWidgetItem *item = new QTableWidgetItem;
      if (i < 4) {
        item->setData(0, QVariant::fromValue(MyPic()));
      } else if (i < 8) {
        QString text("force");
        item->setData(0, QVariant::fromValue(text));
      } else {
        QString text("non-force");
        item->setData(0, QVariant::fromValue(text));
      }
      tableWidget.setItem(i, j, item);
    }
  }
  tableWidget.setItemDelegate(new MyDelegate);
  tableWidget.resize(600, 500);
  tableWidget.show();

  std::cout << "in func main" << std::endl;
  return app.exec();
}
