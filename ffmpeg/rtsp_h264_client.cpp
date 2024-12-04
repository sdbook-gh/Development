#include <cstdio>
#include <string>
#include <fstream>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
}

int main(int argc, char *argv[]) {
  std::string videourl = "rtsp://127.0.0.1:8554/test";
  AVFormatContext *pFormatCtx = nullptr;
  AVDictionary *options = nullptr;

  avformat_network_init();
  //执行网络库的全局初始化。
  //此函数仅用于解决旧版GNUTLS或OpenSSL库的线程安全问题。
  //一旦删除对较旧的GNUTLS和OpenSSL库的支持，此函数将被弃用，并且此函数将不再有任何用途。
  av_dict_set(&options, "buffer_size", "4096000", 0); //设置缓存大小
  av_dict_set(&options, "rtsp_transport", "tcp", 0);  //以tcp的方式打开,
  av_dict_set(&options, "stimeout", "5000000", 0); //设置超时断开链接时间
  av_dict_set(&options, "max_delay", "500000", 0); //设置最大时延
  pFormatCtx =
      avformat_alloc_context(); //用来申请AVFormatContext类型变量并初始化默认参数
  //打开网络流或文件流
  if (avformat_open_input(&pFormatCtx, videourl.c_str(), nullptr, &options) !=
      0) {
    printf("cannot open video\n");
    return -1;
  }
  //获取视频文件信息
  if (avformat_find_stream_info(pFormatCtx, nullptr) < 0) {
    printf("cannot find stream information\n");
    return -1;
  }
  AVDictionaryEntry *tag = nullptr;
  // av_dict_set(&pFormatCtx->metadata, "rotate", "0", 0);
  // //这里可以设置一些属性
  while ((tag = av_dict_get(pFormatCtx->metadata, "", tag, AV_DICT_IGNORE_SUFFIX))) {
    std::string key = tag->key;
    std::string value = tag->value;
    printf("av_dict_get: %s:%s\n", key.c_str(), value.c_str());
  }
  //查找码流中是否有视频流
  int videoindex = -1;
  for (int i = 0; i < pFormatCtx->nb_streams; i++) {
    if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoindex = i;
      break;
    }
  }
  if (videoindex == -1) {
    printf("cannot find video stream\n");
    return -1;
  }
  // 获取视频流的解码器上下文
  const AVCodecParameters *codec_params =
      pFormatCtx->streams[videoindex]->codecpar;
  const AVCodec *codec = avcodec_find_decoder(codec_params->codec_id);
  if (codec == nullptr) {
    printf("cannot find decoder\n");
    return -1;
  }
  AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
  if (codec_ctx == nullptr) {
    printf("cannot allocate video codec context\n");
    return -1;
  }
  // 将编解码器参数复制到解码器上下文
  if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
    printf("cannot copy codec parameters\n");
    return -1;
  }
  // 打开解码器
  if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
    printf("cannot open codec\n");
    return -1;
  }
  AVPacket *av_packet = av_packet_alloc();
  if (av_packet == nullptr) {
    printf("cannot allocate AVPacket\n");
    return -1;
  }
  AVFrame *av_frame = av_frame_alloc();
  if (av_frame == nullptr) {
    printf("cannot allocate AVFrame\n");
    return -1;
  }
  // auto saveYUV420PFrame = [](const std::string &filename, AVFrame *pFrame) {
  //   std::fstream out_file;
  //   out_file.open(filename, std::ios::out | std::ios::binary);
  //   if (out_file) {
  //     out_file.write((char *)pFrame->data[0], pFrame->linesize[0] * pFrame->height);
  //     out_file.write((char *)pFrame->data[1], pFrame->linesize[1] / 2 * pFrame->height);
  //     out_file.write((char *)pFrame->data[2], pFrame->linesize[2] / 2 * pFrame->height);
  //   }
  // };
  // auto saveRGB24Frame = [](const std::string &filename, AVFrame *pFrame) {
  //   std::fstream out_file;
  //   out_file.open(filename, std::ios::out | std::ios::binary);
  //   if (out_file) {
  //     out_file.write((char *)pFrame->data[0], pFrame->linesize[0] * pFrame->height);
  //   }
  // };
  while (true) {
    if (av_read_frame(pFormatCtx, av_packet) >= 0) {
      if (av_packet->stream_index == videoindex) {
        // printf("packet size is %d\n", av_packet->size); // 这里就是接收到的未解码之前的数据
        // 解码视频帧
        if (avcodec_send_packet(codec_ctx, av_packet) == 0) {
          while (avcodec_receive_frame(codec_ctx, av_frame) >= 0) {
            static int save_count{0};
            // if (save_count < 10) {
            //   printf("save frame format %d width %d height %d linesize %d yuv420p size %d\n", (AVPixelFormat)av_frame->format,  av_frame->width, av_frame->height, av_frame->linesize[0], av_image_get_buffer_size(AV_PIX_FMT_YUV420P, av_frame->width, av_frame->height, 1));
            //   saveYUV420PFrame("out_" + std::to_string(save_count) + ".yuv", av_frame);
            //   save_count++;
            // } else {
            //   return 0;
            // }
            // 处理解码后的帧将其转换为RGB格式
            AVFrame *rgb_frame = av_frame_alloc();
            if (rgb_frame == nullptr) {
              printf("cannot allocate AVFrame for rgb\n");
              return -1;
            }
            static int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, av_frame->width, av_frame->height, 1);
            uint8_t *buffer = (uint8_t *)av_malloc(num_bytes * sizeof(uint8_t));
            av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, buffer, AV_PIX_FMT_RGB24, av_frame->width, av_frame->height, 1);
            SwsContext *sws_ctx = sws_getContext(av_frame->width, av_frame->height, (AVPixelFormat)av_frame->format, av_frame->width, av_frame->height, AV_PIX_FMT_RGB24, SWS_BICUBIC, nullptr, nullptr, nullptr);
            if (sws_ctx == nullptr) {
              printf("sws_getContext error\n");
              return -1;
            }
            sws_scale(sws_ctx, (const uint8_t *const *)av_frame->data, av_frame->linesize, 0, av_frame->height, rgb_frame->data, rgb_frame->linesize);
            rgb_frame->width = av_frame->width;
            rgb_frame->height = av_frame->height;
            // if (save_count < 10) {
            //   printf("save frame format %d width %d height %d linesize %d rgb24 size %d\n", (AVPixelFormat)rgb_frame->format,  rgb_frame->width, rgb_frame->height, rgb_frame->linesize[0], av_image_get_buffer_size(AV_PIX_FMT_RGB24, rgb_frame->width, rgb_frame->height, 1));
            //   saveRGB24Frame("out_" + std::to_string(save_count) + ".rgb24", rgb_frame);
            //   save_count++;
            // } else {
            //   return 0;
            // }
            av_frame_free(&rgb_frame);
            av_free(buffer);
            sws_freeContext(sws_ctx);
          }
        }
      }
      av_packet_unref(av_packet);
    } else {
      printf("av_read_frame error\n");
    }
  }
  av_frame_free(&av_frame);
  av_packet_free(&av_packet);
  avcodec_free_context(&codec_ctx);
  avformat_close_input(&pFormatCtx);
  return 0;
}
