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

#endif // MYWIDGET_H
