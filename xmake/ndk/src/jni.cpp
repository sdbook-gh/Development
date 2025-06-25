#include <android/log.h>
#include <jni.h>

#include <string>

#define LOG_TAG "XMAKE_DEMO"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_jniapp_MainActivity_stringFromJNI(JNIEnv* env,
                                                  jobject /* this */) {
  std::string msg = "++++++ Hello from xmake + NDK! ++++++";
  LOGD("%s", msg.c_str());
  return env->NewStringUTF(msg.c_str());
}
