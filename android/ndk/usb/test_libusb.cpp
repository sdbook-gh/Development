#include <jni.h>
#include <android/log.h>
#include <libusb.h>

#define TAG "UsbSample"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/* 端点地址必须与设备描述符一致 */
#define BULK_EP_OUT 0x01
#define BULK_EP_IN 0x81
#define BUF_LEN 64

static libusb_context* ctx = nullptr;
static libusb_device_handle* device_handle = nullptr;

extern "C" JNIEXPORT jboolean JNICALL Java_com_example_jniapp_MainActivity_openusb(JNIEnv* env, jclass clazz, jint fd, jint vid, jint pid) {
  LOGI("开始打开USB设备, FD: %d, VID: 0x%04X, PID: 0x%04X", fd, vid, pid);

  // 初始化libusb
  int result = libusb_init(&ctx);
  if (result < 0) {
    LOGE("libusb初始化失败: %s", libusb_error_name(result));
    return JNI_FALSE;
  }

  // 设置调试级别
  libusb_set_debug(ctx, LIBUSB_LOG_LEVEL_INFO);

  // 使用Android的文件描述符包装libusb设备
  result = libusb_wrap_sys_device(ctx, (intptr_t)fd, &device_handle);
  if (result < 0) {
    LOGE("libusb_wrap_sys_device失败: %s", libusb_error_name(result));
    libusb_exit(ctx);
    ctx = nullptr;
    return JNI_FALSE;
  }

  // 验证设备VID/PID
  libusb_device* device = libusb_get_device(device_handle);
  struct libusb_device_descriptor desc;
  result = libusb_get_device_descriptor(device, &desc);
  if (result < 0) {
    LOGE("获取设备描述符失败: %s", libusb_error_name(result));
    libusb_close(device_handle);
    device_handle = nullptr;
    libusb_exit(ctx);
    ctx = nullptr;
    return JNI_FALSE;
  }

  LOGI("设备VID: 0x%04X, PID: 0x%04X", desc.idVendor, desc.idProduct);

  if (desc.idVendor != vid || desc.idProduct != pid) {
    LOGE("设备VID/PID不匹配");
    libusb_close(device_handle);
    device_handle = nullptr;
    libusb_exit(ctx);
    ctx = nullptr;
    return JNI_FALSE;
  }

  // 声明接口
  result = libusb_claim_interface(device_handle, 0);
  if (result < 0) {
    LOGE("声明接口失败: %s", libusb_error_name(result));
    libusb_close(device_handle);
    device_handle = nullptr;
    libusb_exit(ctx);
    ctx = nullptr;
    return JNI_FALSE;
  }

  LOGI("USB设备打开成功");
  return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL Java_com_example_jniapp_MainActivity_closeusb(JNIEnv* env, jclass clazz) {
  LOGI("关闭USB设备");

  if (device_handle) {
    libusb_release_interface(device_handle, 0);
    libusb_close(device_handle);
    device_handle = nullptr;
  }

  if (ctx) {
    libusb_exit(ctx);
    ctx = nullptr;
  }
}
