#include <GLES2/gl2.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define LOG_TAG "native-lib"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

static GLuint textureId;
static GLuint program;
static int viewWidth, viewHeight;

// 顶点和纹理坐标
const GLfloat vertices[] = {
  -1.0f, 1.0f, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f, 1.0f,
};

// 简易着色器
const char* vShaderStr = "attribute vec4 aPosition;\n"
                         "attribute vec2 aTexCoord;\n"
                         "varying vec2 vTexCoord;\n"
                         "void main() {\n"
                         "  gl_Position = aPosition;\n"
                         "  vTexCoord = aTexCoord;\n"
                         "}\n";

const char* fShaderStr = "precision mediump float;\n"
                         "varying vec2 vTexCoord;\n"
                         "uniform sampler2D sTexture;\n"
                         "void main() {\n"
                         "  gl_FragColor = texture2D(sTexture, vTexCoord);\n"
                         "}\n";

// 编译着色器
static GLuint loadShader(GLenum type, const char* shaderSrc) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &shaderSrc, nullptr);
  glCompileShader(shader);
  return shader;
}

// 创建程序
static void initProgram() {
  GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vShaderStr);
  GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fShaderStr);
  program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);
}

// 从 asset 中加载 JPEG，解码并生成 OpenGL 纹理
static void loadTextureFromAsset(AAssetManager* mgr) {
  AAsset* asset = AAssetManager_open(mgr, "image.jpg", AASSET_MODE_UNKNOWN);
  off_t length = AAsset_getLength(asset);
  const void* data = AAsset_getBuffer(asset);

  int width, height, channels;
  unsigned char* pixels = stbi_load_from_memory((const stbi_uc*)data, (int)length, &width, &height, &channels, 4);
  glGenTextures(1, &textureId);
  glBindTexture(GL_TEXTURE_2D, textureId);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  stbi_image_free(pixels);
  AAsset_close(asset);
  LOGI("Loaded texture %d (%dx%d)", textureId, width, height);
}

// JNI: Surface 创建时调用
extern "C" JNIEXPORT void JNICALL Java_com_example_jniapp_MyRenderer_nativeOnSurfaceCreated(JNIEnv* env, jobject /* this */, jobject assetManager) {
  AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
  initProgram();
  loadTextureFromAsset(mgr);
  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
}

// JNI: Surface 大小变化时调用
extern "C" JNIEXPORT void JNICALL Java_com_example_jniapp_MyRenderer_nativeOnSurfaceChanged(JNIEnv* env, jobject /* this */, jint width, jint height) {
  viewWidth = width;
  viewHeight = height;
  glViewport(0, 0, width, height);
}

// JNI: 每帧渲染
extern "C" JNIEXPORT void JNICALL Java_com_example_jniapp_MyRenderer_nativeOnDrawFrame(JNIEnv* env, jobject /* this */) {
  glClear(GL_COLOR_BUFFER_BIT);
  glUseProgram(program);

  GLint posLoc = glGetAttribLocation(program, "aPosition");
  GLint texLoc = glGetAttribLocation(program, "aTexCoord");
  GLint samplerLoc = glGetUniformLocation(program, "sTexture");

  glEnableVertexAttribArray(posLoc);
  glEnableVertexAttribArray(texLoc);
  glVertexAttribPointer(posLoc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), vertices);
  glVertexAttribPointer(texLoc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), vertices + 2);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureId);
  glUniform1i(samplerLoc, 0);

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glDisableVertexAttribArray(posLoc);
  glDisableVertexAttribArray(texLoc);
}
