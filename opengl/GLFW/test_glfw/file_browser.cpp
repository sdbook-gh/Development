#include <GL/gl.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

// 文件项结构
struct FileItem {
  std::string name;
  std::string fullPath;
  bool isDirectory;
  uintmax_t size;
  std::filesystem::file_time_type lastModified;

  FileItem(const std::string& n, const std::string& path, bool isDir, uintmax_t s, std::filesystem::file_time_type time) : name(n), fullPath(path), isDirectory(isDir), size(s), lastModified(time) {}
};

// 简单的GUI组件基类
class Widget {
public:
  float x, y, width, height;
  bool visible = true;

  Widget(float x, float y, float w, float h) : x(x), y(y), width(w), height(h) {}
  virtual ~Widget() = default;
  virtual void render() = 0;
  virtual bool handleClick(double mx, double my) { return false; }
  virtual bool handleKey(int key, int action) { return false; }

  bool contains(double px, double py) const { return px >= x && px <= x + width && py >= y && py <= y + height; }
};

// 文本渲染辅助函数
void renderText(const std::string& text, float x, float y, float scale = 1.0f) {
  glRasterPos2f(x, y);
  for (char c : text) { glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c); }
}

// 矩形渲染辅助函数
void renderRect(float x, float y, float width, float height, float r, float g, float b, float a = 1.0f) {
  glColor4f(r, g, b, a);
  glBegin(GL_QUADS);
  glVertex2f(x, y);
  glVertex2f(x + width, y);
  glVertex2f(x + width, y + height);
  glVertex2f(x, y + height);
  glEnd();
}

// 线条渲染辅助函数
void renderLine(float x1, float y1, float x2, float y2, float r, float g, float b) {
  glColor3f(r, g, b);
  glBegin(GL_LINES);
  glVertex2f(x1, y1);
  glVertex2f(x2, y2);
  glEnd();
}

// 地址栏组件
class AddressBar : public Widget {
public:
  std::string currentPath;
  bool editing = false;
  std::string editText;

  AddressBar(float x, float y, float w, float h) : Widget(x, y, w, h) {
    currentPath = fs::current_path().string();
    editText = currentPath;
  }

  void setPath(const std::string& path) {
    currentPath = path;
    editText = path;
  }

  void render() override {
    // 背景
    renderRect(x, y, width, height, 1.0f, 1.0f, 1.0f, 1.0f);

    // 边框
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y + height);
    glVertex2f(x, y + height);
    glEnd();

    // 文本
    glColor3f(0.0f, 0.0f, 0.0f);
    std::string displayText = editing ? editText : currentPath;
    if (displayText.length() > 80) { displayText = "..." + displayText.substr(displayText.length() - 77); }
    renderText(displayText, x + 5, y + height - 15);
  }

  bool handleClick(double mx, double my) override {
    if (contains(mx, my)) {
      editing = true;
      return true;
    } else {
      editing = false;
      return false;
    }
  }

  bool handleKey(int key, int action) override {
    if (!editing) return false;

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
      if (key == GLFW_KEY_ENTER) {
        currentPath = editText;
        editing = false;
        return true;
      } else if (key == GLFW_KEY_ESCAPE) {
        editText = currentPath;
        editing = false;
        return true;
      } else if (key == GLFW_KEY_BACKSPACE && !editText.empty()) {
        editText.pop_back();
        return true;
      }
    }
    return false;
  }

  void addChar(char c) {
    if (editing) { editText += c; }
  }
};

// 文件列表组件
class FileList : public Widget {
public:
  std::vector<FileItem> files;
  int selectedIndex = -1;
  int scrollOffset = 0;
  float itemHeight = 40.0f;
  std::string currentPath;

  FileList(float x, float y, float w, float h) : Widget(x, y, w, h) { loadDirectory(fs::current_path().string()); }

  void loadDirectory(const std::string& path) {
    files.clear();
    selectedIndex = -1;
    scrollOffset = 0;
    currentPath = path;

    try {
      // 添加返回上级目录项
      if (path != fs::path(path).root_path()) { files.emplace_back("..", fs::path(path).parent_path().string(), true, 0, std::filesystem::file_time_type{}); }

      // 遍历目录
      std::vector<FileItem> directories;
      std::vector<FileItem> regularFiles;

      for (const auto& entry : fs::directory_iterator(path)) {
        try {
          std::string name = entry.path().filename().string();
          std::string fullPath = entry.path().string();
          bool isDir = entry.is_directory();
          uintmax_t size = isDir ? 0 : (entry.exists() ? fs::file_size(entry.path()) : 0);
          auto lastModified = entry.last_write_time();

          FileItem item(name, fullPath, isDir, size, lastModified);

          if (isDir) {
            directories.push_back(item);
          } else {
            regularFiles.push_back(item);
          }
        } catch (const std::exception&) {
          // 跳过无法访问的文件
          continue;
        }
      }

      // 排序
      std::sort(directories.begin(), directories.end(), [](const FileItem& a, const FileItem& b) { return a.name < b.name; });
      std::sort(regularFiles.begin(), regularFiles.end(), [](const FileItem& a, const FileItem& b) { return a.name < b.name; });

      // 添加到文件列表
      files.insert(files.end(), directories.begin(), directories.end());
      files.insert(files.end(), regularFiles.begin(), regularFiles.end());

    } catch (const std::exception& e) { std::cerr << "Error loading directory: " << e.what() << std::endl; }
  }

  std::string formatFileSize(uintmax_t size) {
    if (size == 0) return "";

    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double sizeD = static_cast<double>(size);

    while (sizeD >= 1024.0 && unitIndex < 4) {
      sizeD /= 1024.0;
      unitIndex++;
    }

    std::ostringstream oss;
    if (unitIndex == 0) {
      oss << static_cast<uintmax_t>(sizeD) << " " << units[unitIndex];
    } else {
      oss << std::fixed << std::setprecision(1) << sizeD << " " << units[unitIndex];
    }
    return oss.str();
  }

  void render() override {
    // 背景
    renderRect(x, y, width, height, 1.0f, 1.0f, 1.0f, 1.0f);

    // 边框
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y + height);
    glVertex2f(x, y + height);
    glEnd();

    // 表头
    float headerY = y + height - 25;
    renderRect(x, headerY, width, 25, 0.95f, 0.95f, 0.95f, 1.0f);

    glColor3f(0.0f, 0.0f, 0.0f);
    renderText("Name", x + 10, headerY + 15);
    renderText("Size", x + width - 200, headerY + 15);
    renderText("Modified", x + width - 100, headerY + 15);

    // 表头分隔线
    renderLine(x, headerY, x + width, headerY, 0.7f, 0.7f, 0.7f);

    // 文件列表
    int visibleItems = static_cast<int>((height - 25) / itemHeight);
    int startIndex = scrollOffset;
    int endIndex = std::min(startIndex + visibleItems, static_cast<int>(files.size()));

    for (int i = startIndex; i < endIndex; i++) {
      float itemY = y + height - 25 - (i - startIndex + 1) * itemHeight;

      // 选中高亮
      if (i == selectedIndex) { renderRect(x + 1, itemY, width - 2, itemHeight, 0.3f, 0.6f, 1.0f, 0.3f); }

      // 文件图标和名称
      glColor3f(0.0f, 0.0f, 0.0f);
      std::string icon = files[i].isDirectory ? "folder: " : "file: ";
      renderText(icon + files[i].name, x + 10, itemY + 15);

      // 文件大小
      if (!files[i].isDirectory) {
        std::string sizeStr = formatFileSize(files[i].size);
        renderText(sizeStr, x + width - 200, itemY + 15);
      }

      // 分隔线
      if (i < endIndex - 1) { renderLine(x + 5, itemY, x + width - 5, itemY, 0.9f, 0.9f, 0.9f); }
    }
  }

  bool handleClick(double mx, double my) override {
    if (!contains(mx, my)) return false;

    float headerY = y + height - 25;
    if (my > headerY) return true; // 点击表头

    int clickedIndex = static_cast<int>((headerY - my) / itemHeight) + scrollOffset;

    if (clickedIndex >= 0 && clickedIndex < files.size()) {
      if (clickedIndex == selectedIndex) {
        // 双击效果
        if (files[clickedIndex].isDirectory) {
          loadDirectory(files[clickedIndex].fullPath);
          return true;
        }
      } else {
        selectedIndex = clickedIndex;
      }
    }

    return true;
  }

  void scroll(double deltaY) {
    int maxScroll = std::max(0, static_cast<int>(files.size()) - static_cast<int>((height - 25) / itemHeight));
    scrollOffset = std::max(0, std::min(maxScroll, scrollOffset + static_cast<int>(deltaY)));
  }

  std::string getSelectedPath() const {
    if (selectedIndex >= 0 && selectedIndex < files.size()) { return files[selectedIndex].fullPath; }
    return "";
  }
};

// 目录树组件
class DirectoryTree : public Widget {
public:
  struct TreeNode {
    std::string name;
    std::string fullPath;
    bool expanded = false;
    int level = 0;
    std::vector<std::unique_ptr<TreeNode>> children;

    TreeNode(const std::string& n, const std::string& path, int l) : name(n), fullPath(path), level(l) {}
  };

  std::unique_ptr<TreeNode> root;
  int selectedIndex = -1;
  int scrollOffset = 0;
  float itemHeight = 18.0f;
  std::vector<TreeNode*> flatList;

  DirectoryTree(float x, float y, float w, float h) : Widget(x, y, w, h) { buildTree(); }

  void buildTree() {
#ifdef _WIN32
    // Windows: 显示驱动器
    root = std::make_unique<TreeNode>("This PC", "", 0);
    DWORD drives = GetLogicalDrives();
    for (int i = 0; i < 26; i++) {
      if (drives & (1 << i)) {
        std::string driveLetter = std::string(1, 'A' + i) + ":\\";
        auto driveNode = std::make_unique<TreeNode>(driveLetter, driveLetter, 1);
        root->children.push_back(std::move(driveNode));
      }
    }
#else
    // Linux/Unix: 从根目录开始
    root = std::make_unique<TreeNode>("/", "/", 0);
    loadChildren(root.get());
#endif
    flattenTree();
  }

  void loadChildren(TreeNode* node) {
    if (node->children.empty()) {
      try {
        for (const auto& entry : fs::directory_iterator(node->fullPath)) {
          if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            std::string fullPath = entry.path().string();
            auto child = std::make_unique<TreeNode>(name, fullPath, node->level + 1);
            node->children.push_back(std::move(child));
          }
        }
        std::sort(node->children.begin(), node->children.end(), [](const auto& a, const auto& b) { return a->name < b->name; });
      } catch (const std::exception&) {
        // 忽略无法访问的目录
      }
    }
  }

  void flattenTree() {
    flatList.clear();
    flattenNode(root.get());
  }

  void flattenNode(TreeNode* node) {
    if (node->level > 0) { // 不显示根节点
      flatList.push_back(node);
    }

    if (node->expanded) {
      loadChildren(node);
      for (const auto& child : node->children) { flattenNode(child.get()); }
    }
  }

  void render() override {
    // 背景
    renderRect(x, y, width, height, 0.98f, 0.98f, 0.98f, 1.0f);

    // 边框
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y + height);
    glVertex2f(x, y + height);
    glEnd();

    // 树节点
    int visibleItems = static_cast<int>(height / itemHeight);
    int startIndex = scrollOffset;
    int endIndex = std::min(startIndex + visibleItems, static_cast<int>(flatList.size()));

    for (int i = startIndex; i < endIndex; i++) {
      TreeNode* node = flatList[i];
      float itemY = y + height - (i - startIndex + 1) * itemHeight;

      // 选中高亮
      if (i == selectedIndex) { renderRect(x + 1, itemY, width - 2, itemHeight, 0.3f, 0.6f, 1.0f, 0.3f); }

      // 缩进
      float indentX = x + 5 + node->level * 15;

      // 展开/折叠图标
      glColor3f(0.0f, 0.0f, 0.0f);
      if (!node->children.empty() || node->level == 1) {
        std::string expandIcon = node->expanded ? "▼ " : "▶ ";
        renderText(expandIcon, indentX, itemY + 13);
        indentX += 20;
      } else {
        indentX += 10;
      }

      // 文件夹图标和名称
      renderText("📁 " + node->name, indentX, itemY + 13);
    }
  }

  bool handleClick(double mx, double my) override {
    if (!contains(mx, my)) return false;

    int clickedIndex = static_cast<int>((y + height - my) / itemHeight) + scrollOffset;

    if (clickedIndex >= 0 && clickedIndex < flatList.size()) {
      TreeNode* node = flatList[clickedIndex];

      // 检查是否点击展开图标
      float indentX = x + 5 + node->level * 15;
      if (mx < indentX + 20 && (!node->children.empty() || node->level == 1)) {
        node->expanded = !node->expanded;
        flattenTree();
        return true;
      }

      selectedIndex = clickedIndex;
      return true;
    }

    return false;
  }

  void scroll(double deltaY) {
    int maxScroll = std::max(0, static_cast<int>(flatList.size()) - static_cast<int>(height / itemHeight));
    scrollOffset = std::max(0, std::min(maxScroll, scrollOffset + static_cast<int>(deltaY)));
  }

  std::string getSelectedPath() const {
    if (selectedIndex >= 0 && selectedIndex < flatList.size()) { return flatList[selectedIndex]->fullPath; }
    return "";
  }
};

// 主应用类
class FileExplorer {
public:
  GLFWwindow* window;
  AddressBar* addressBar;
  FileList* fileList;
  DirectoryTree* directoryTree;
  int windowWidth = 400;
  int windowHeight = 300;

  FileExplorer() {
    initGLFW();
    setupWidgets();
  }

  ~FileExplorer() {
    delete addressBar;
    delete fileList;
    delete directoryTree;
    glfwTerminate();
  }

  void initGLFW() {
    if (!glfwInit()) { throw std::runtime_error("Failed to initialize GLFW"); }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    window = glfwCreateWindow(windowWidth, windowHeight, "File Explorer", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);

    // 设置回调
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCharCallback(window, charCallback);

    // OpenGL设置
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    updateProjection();
  }

  void setupWidgets() {
    float treeWidth = 250.0f;
    float addressHeight = 30.0f;

    addressBar = new AddressBar(treeWidth + 10, windowHeight - addressHeight - 10, windowWidth - treeWidth - 20, addressHeight);

    fileList = new FileList(treeWidth + 10, 10, windowWidth - treeWidth - 20, windowHeight - addressHeight - 30);

    directoryTree = new DirectoryTree(10, 10, treeWidth, windowHeight - 20);
  }

  void updateProjection() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
  }

  void run() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      // 渲染组件
      directoryTree->render();
      fileList->render();
      addressBar->render();

      glfwSwapBuffers(window);
    }
  }

  static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    FileExplorer* app = static_cast<FileExplorer*>(glfwGetWindowUserPointer(window));
    app->windowWidth = width;
    app->windowHeight = height;
    glViewport(0, 0, width, height);
    app->updateProjection();

    // 更新组件尺寸
    float treeWidth = 250.0f;
    float addressHeight = 30.0f;

    app->addressBar->x = treeWidth + 10;
    app->addressBar->y = height - addressHeight - 10;
    app->addressBar->width = width - treeWidth - 20;

    app->fileList->x = treeWidth + 10;
    app->fileList->width = width - treeWidth - 20;
    app->fileList->height = height - addressHeight - 30;

    app->directoryTree->height = height - 20;
  }

  static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
      FileExplorer* app = static_cast<FileExplorer*>(glfwGetWindowUserPointer(window));
      double xpos, ypos;
      glfwGetCursorPos(window, &xpos, &ypos);
      ypos = app->windowHeight - ypos; // 翻转Y坐标

      // 检查组件点击
      if (app->directoryTree->handleClick(xpos, ypos)) {
        std::string path = app->directoryTree->getSelectedPath();
        if (!path.empty()) {
          app->fileList->loadDirectory(path);
          app->addressBar->setPath(path);
        }
      } else if (app->fileList->handleClick(xpos, ypos)) {
        // 文件列表点击已在组件内处理
      } else if (app->addressBar->handleClick(xpos, ypos)) {
        // 地址栏点击已在组件内处理
      }
    }
  }

  static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    FileExplorer* app = static_cast<FileExplorer*>(glfwGetWindowUserPointer(window));
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    ypos = app->windowHeight - ypos;

    if (app->directoryTree->contains(xpos, ypos)) {
      app->directoryTree->scroll(-yoffset);
    } else if (app->fileList->contains(xpos, ypos)) {
      app->fileList->scroll(-yoffset);
    }
  }

  static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    FileExplorer* app = static_cast<FileExplorer*>(glfwGetWindowUserPointer(window));

    if (app->addressBar->handleKey(key, action)) {
      if (key == GLFW_KEY_ENTER && action == GLFW_PRESS) {
        // 地址栏回车，加载新目录
        try {
          app->fileList->loadDirectory(app->addressBar->currentPath);
        } catch (const std::exception& e) {
          std::cerr << "Cannot access path: " << e.what() << std::endl;
          app->addressBar->editText = app->fileList->currentPath;
        }
      }
    }

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) { glfwSetWindowShouldClose(window, GLFW_TRUE); }
  }

  static void charCallback(GLFWwindow* window, unsigned int codepoint) {
    FileExplorer* app = static_cast<FileExplorer*>(glfwGetWindowUserPointer(window));
    if (codepoint < 128) { // 简单ASCII字符
      app->addressBar->addChar(static_cast<char>(codepoint));
    }
  }
};

int main(int argc, char* argv[]) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(400, 300);
  try {
    FileExplorer app;
    app.run();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
