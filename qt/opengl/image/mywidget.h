#ifndef MYWIDGET_H
#define MYWIDGET_H

#include <QObject>
#include <QOpenGLWidget>
#include <GL/gl.h>
#include <GL/glu.h>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QGLWidget>
#include <QImage>

class MyGLWidget : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT
  public:
    explicit MyGLWidget(QWidget *parent = 0);

  signals:

  public slots:
    void initializeGL() Q_DECL_OVERRIDE;
    void resizeGL(int w, int h) Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    void setImage(const QImage &image);
    void initTextures();
    void initShaders();

  private:
    QVector<QVector3D> vertices;
    QVector<QVector2D> texCoords;
    QOpenGLShaderProgram program;
    QOpenGLTexture *texture;
    QMatrix4x4 projection;
};

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QImage>
#include <QGLWidget>
#include <QMouseEvent>
#include <QPainter>
#include <QDebug>

// class ImgShow : public QOpenGLWidget, protected QOpenGLFunctions {
//   Q_OBJECT

//   public:
//     ImgShow(QWidget *parent = nullptr)
//         : QOpenGLWidget(parent) {}

//   protected:
//     void initializeGL() override {
//         initializeOpenGLFunctions();
//         glClearColor(0.0, 0.0, 0.0, 1.0);
//         // 加载图片并生成纹理
//         QImage image("/apollo/modules/tools/visualizer/images/no_image.png");  // 替换为你的图片路径
//         if (image.isNull()) {
//             qDebug() << "Image load failed!";
//             return;
//         }
//         texture = new QOpenGLTexture(image, QOpenGLTexture::DontGenerateMipMaps);
//         texture->setMinificationFilter(QOpenGLTexture::Nearest);
//         texture->setMagnificationFilter(QOpenGLTexture::Nearest);
//     }

//     void paintGL() override {
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//         glEnable(GL_TEXTURE_2D);
//         texture->bind();
//         // 绘制纹理为全屏四边形
//         glBegin(GL_QUADS);
//         glTexCoord2f(0, 1 - 0);
//         glVertex2f(-1, -1);
//         glTexCoord2f(1, 1 - 0);
//         glVertex2f(1, -1);
//         glTexCoord2f(1, 1 - 1);
//         glVertex2f(1, 1);
//         glTexCoord2f(0, 1 - 1);
//         glVertex2f(-1, 1);
//         glEnd();
//         glDisable(GL_TEXTURE_2D);
//     }

//     void resizeGL(int w, int h) override {
//         glViewport(0, 0, w, h);
//     }

//   private:
//     QOpenGLTexture *texture = nullptr;
// };

#endif  // MYWIDGET_H
