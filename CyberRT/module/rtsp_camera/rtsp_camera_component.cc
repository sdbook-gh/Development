/******************************************************************************
 * @file rtsp_camera_component.cc
 *****************************************************************************/

#include "module/rtsp_camera/rtsp_camera_component.h"

#include <cstdio>
#include <string>
#include <fstream>
#include <thread>
#include <chrono>
#include <atomic>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include <gflags/gflags.h>
#include "modules/common_msgs/sensor_msgs/sensor_image.pb.h"

// #define DEBUG 1

DEFINE_string(rtsp_url, "rtsp://127.0.0.1:8554/rtsp_camera", "rtsp url");
DEFINE_string(output_topic, "/camera/rtsp_camera", "output topic");

namespace apollo {

bool RtspCamera::Init() {
  AINFO << "rtsp_url:" << FLAGS_rtsp_url;
  AINFO << "output_topic:" << FLAGS_output_topic;
  std::thread rtsp_camera_thread([this]{
    while(run()) {
      cyber::SleepFor(std::chrono::seconds(10));
    }
  });
  rtsp_camera_thread.detach();
  service = node_->CreateService<apollo::RtspCameraTriggerRequest, apollo::RtspCameraTriggerResponse>(
    "rtsp_camera",
    [](const auto& request, auto& response) {
        AINFO << request->action();
        response->set_result("ok");
    }
  );
  AINFO << "Init RtspCamera succedded.";
  return true;
}

int RtspCamera::run() {
  avformat_network_init();
  AVDictionary *options{nullptr};
  av_dict_set(&options, "rtsp_transport", "tcp", 0);
  AVFormatContext* format_context = avformat_alloc_context();
  if (format_context == nullptr) {
    AERROR << "avformat_alloc_context error";
    return -1;
  }
  if (avformat_open_input(&format_context, FLAGS_rtsp_url.c_str(), nullptr, &options) != 0) {
    AERROR << "avformat_open_input error";
    avformat_close_input(&format_context);
    return -1;
  }
  if (avformat_find_stream_info(format_context, nullptr) < 0) {
    AERROR << "avformat_find_stream_info error";
    avformat_close_input(&format_context);
    return -1;
  }
  int videoindex{-1};
  for (size_t i = 0; i < format_context->nb_streams; i++) {
    if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoindex = i;
      break;
    }
  }
  if (videoindex == -1) {
    AERROR << "cannot find video";
    avformat_close_input(&format_context);
    return -1;
  }
  const AVCodecParameters *codec_params = format_context->streams[videoindex]->codecpar;
  const AVCodec *codec = avcodec_find_decoder(codec_params->codec_id);
  if (codec == nullptr) {
    AERROR << "avcodec_find_decoder error";
    avformat_close_input(&format_context);
    return -1;
  }
  AVCodecContext *codec_context = avcodec_alloc_context3(codec);
  if (codec_context == nullptr) {
    AERROR << "avcodec_alloc_context3 error";
    avformat_close_input(&format_context);
    return -1;
  }
  if (avcodec_parameters_to_context(codec_context, codec_params) < 0) {
    AERROR << "avcodec_parameters_to_context error";
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);
    return -1;
  }
  if (avcodec_open2(codec_context, codec, nullptr) < 0) {
    AERROR << "avcodec_open2 error";
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);
    return -1;
  }
  codec_context->thread_count = 4;
  AVPacket *av_packet = av_packet_alloc();
  if (av_packet == nullptr) {
    AERROR << "av_packet_alloc error";
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);
    return -1;
  }
  AVFrame *av_frame = av_frame_alloc();
  if (av_frame == nullptr) {
    AERROR << "av_frame_alloc error";
    av_packet_free(&av_packet);
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);
    return -1;
  }
  #ifdef DEBUG
  auto saveRGB24Frame = [](const std::string &filename, AVFrame *pFrame) {
    std::fstream out_file;
    out_file.open(filename, std::ios::out | std::ios::binary);
    if (out_file) {
      out_file.write((char *)pFrame->data[0], pFrame->linesize[0] * pFrame->height);
    }
  };
  #endif
  auto topic_writer = node_->CreateWriter<apollo::drivers::Image>(FLAGS_output_topic);
  while (!cyber::IsShutdown()) {
    if (av_read_frame(format_context, av_packet) >= 0) {
      if (av_packet->stream_index == videoindex) {
        if (avcodec_send_packet(codec_context, av_packet) == 0) {
          while (avcodec_receive_frame(codec_context, av_frame) >= 0) {
            static int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, av_frame->width, av_frame->height, 1);
            static uint8_t *buffer = (uint8_t *)av_malloc(num_bytes * sizeof(uint8_t));
            if (buffer == nullptr) {
              AERROR << "av_malloc error";
              continue;
            }
            AVFrame *rgb_frame = av_frame_alloc();
            if (rgb_frame == nullptr) {
              AERROR << "av_packet_alloc error";
              continue;
            }
            av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, buffer, AV_PIX_FMT_RGB24, av_frame->width, av_frame->height, 1);
            SwsContext *sws_ctx = sws_getContext(av_frame->width, av_frame->height, (AVPixelFormat)av_frame->format, av_frame->width, av_frame->height, AV_PIX_FMT_RGB24, SWS_BICUBIC, nullptr, nullptr, nullptr);
            if (sws_ctx == nullptr) {
              AERROR << "sws_getContext error";
              av_frame_free(&rgb_frame);
              continue;
            }
            sws_scale(sws_ctx, av_frame->data, av_frame->linesize, 0, av_frame->height, rgb_frame->data, rgb_frame->linesize);
            rgb_frame->format = AV_PIX_FMT_RGB24;
            rgb_frame->width = av_frame->width;
            rgb_frame->height = av_frame->height;
            #ifdef DEBUG
            static int save_count{0};
            if (save_count < 10) {
              saveRGB24Frame("out_" + std::to_string(save_count) + ".rgb24", rgb_frame);
              save_count++;
            }
            #endif
            apollo::drivers::Image out_image;
            out_image.mutable_header()->set_frame_id("rtsp_camera");
            out_image.set_measurement_time(cyber::Time::Now().ToNanosecond());
            topic_writer->Write(out_image);
            av_frame_free(&rgb_frame);
            sws_freeContext(sws_ctx);
            av_frame_unref(av_frame);
          }
        } else {
          AERROR << "avcodec_send_packet error";
        }
      }
      av_packet_unref(av_packet);
    } else {
      AERROR << "av_read_frame error";
    }
  }
  av_frame_free(&av_frame);
  av_packet_free(&av_packet);
  avcodec_free_context(&codec_context);
  avformat_close_input(&format_context);
  return 0;
}

} // namespace apollo
