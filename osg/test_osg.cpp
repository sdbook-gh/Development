#include <iostream>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>

osg::ref_ptr<osg::Node> createScene() {
  osg::ref_ptr<osg::Group> root = new osg::Group;
  osg::ref_ptr<osg::Geode> geode = new osg::Geode;
  root->addChild(geode);
  osg::ref_ptr<osg::Shape> sphere =
      new osg::Sphere(osg::Vec3(-5.f, 0.f, 0.f), 3.f);
  osg::ref_ptr<osg::ShapeDrawable> sphereDrawable =
      new osg::ShapeDrawable(sphere);
  sphereDrawable->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
  geode->addDrawable(sphereDrawable);
  osg::ref_ptr<osg::Shape> box = new osg::Box(osg::Vec3(5.f, 0.f, 0.f), 4.f);
  osg::ref_ptr<osg::ShapeDrawable> boxDrawable = new osg::ShapeDrawable(box);
  boxDrawable->setColor(osg::Vec4(0.f, 0.f, 1.f, 1.f));
  geode->addDrawable(boxDrawable);
  return root;
}

class MyHandler : public osgGA::GUIEventHandler {
public:
  MyHandler(osg::ref_ptr<osg::Group> parent) : _parent(parent) {
    _root = new osg::Group;
    if (_parent.valid()) {
      _parent->addChild(_root);
    }
  }

  virtual bool handle(const osgGA::GUIEventAdapter &ea,
                      osgGA::GUIActionAdapter &aa,
                      osg::Object *,
                      osg::NodeVisitor *) override {
    if ((ea.getEventType() & osgGA::GUIEventAdapter::EventType::PUSH) &&
        (ea.getModKeyMask() & osgGA::GUIEventAdapter::ModKeyMask::MODKEY_CTRL) &&
        (ea.getButton() & osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON)) {
      auto view = aa.asView();
      if (!view) {
        return false;
      }
      auto camera = view->getCamera();
      if (!camera) {
        return false;
      }
      const auto &viewMatrix = camera->getViewMatrix();
      const auto &projMatrix = camera->getProjectionMatrix();
      auto vpMatrix = viewMatrix * projMatrix;
      auto invvpMatrix = osg::Matrixd::inverse(vpMatrix);
      osg::Vec3d eye, center, up;
      viewMatrix.getLookAt(eye, center, up);
      osg::Vec4d ndcFar(ea.getXnormalized(), ea.getYnormalized(), 1.f, 1.f);
      auto worldFar = invvpMatrix.preMult(ndcFar);
      worldFar /= worldFar.w();
      createOrUpdateRay(eye, osg::Vec3d(worldFar.x(), worldFar.y(), worldFar.z()));
      return true;
    }
    return false;
  }

  void createOrUpdateRay(const osg::Vec3d &start, const osg::Vec3d &end) {
    if (!_rayNode && _root) {
      // create new ray node
      _rayNode = new osg::Geode;
      _root->addChild(_rayNode);
      osg::ref_ptr<osg::Geometry> geom = new osg::Geometry;
      _rayNode->addDrawable(geom);
      geom->setUseVertexBufferObjects(true);
      _rayVertices = new osg::Vec3Array(2);
      geom->setVertexArray(_rayVertices);
      osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array(2);
      colors->at(0) = osg::Vec4(1.f, 0.f, 0.f, 1.f);
      colors->at(1) = osg::Vec4(1.f, 1.f, 0.f, 1.f);
      geom->setColorArray(colors, osg::Array::Binding::BIND_PER_VERTEX);
      geom->addPrimitiveSet(
          new osg::DrawArrays(GL_LINES, 0, _rayVertices->size()));
      geom->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    }
    if (_rayVertices) {
      _rayVertices->at(0) = start;
      _rayVertices->at(1) = end;
      _rayVertices->dirty();
    }
  }

private:
  osg::ref_ptr<osg::Geode> _rayNode;
  osg::observer_ptr<osg::Group> _parent;
  osg::ref_ptr<osg::Vec3Array> _rayVertices;
  osg::ref_ptr<osg::Group> _root;
};

int main(int argc, char **argv) {
  /*std::cout << "Hello, osg!" << std::endl;
  osg::ref_ptr<osgViewer::Viewer> viewer = new osgViewer::Viewer;
  viewer->setUpViewInWindow(50, 50, 800, 600);
  osg::ref_ptr<osg::Geode> geode = new osg::Geode;
  geode->addDrawable(
      new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(), 1.0f)));
  viewer->setSceneData(geode);
  osg::ref_ptr<osg::Texture> texture =
      new osg::Texture2D(osgDB::readRefImageFile("../test.png"));
  auto stateSet = geode->getOrCreateStateSet();
  stateSet->setTextureAttributeAndModes(0, texture);
  return viewer->run();*/

  osgViewer::Viewer viewer;
  viewer.setUpViewInWindow(100, 100, 800, 600);
  osg::ref_ptr<osg::Group> root = new osg::Group;
  root->addChild(createScene());
  viewer.setSceneData(root);
  viewer.addEventHandler(new MyHandler(root));
  return viewer.run();
}
