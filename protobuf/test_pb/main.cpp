#include "amessage.v2.pb-c.h"
#include "amessage.pb.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iosfwd>
#include <vector>
#include <cfloat>

int main(int argc, const char *argv[]) {
  float a = 0;
  // {
  //   test::protobuf::cpp::PointCloud pointCloud;
  //   pointCloud.set_height(1080);
  //   pointCloud.set_width(1920);
  //   pointCloud.set_point_step(1);
  //   pointCloud.set_row_step(1);
  //   for (int i = 0; i < 1920 * 1080; i++) {
  //     auto *point = pointCloud.add_points();
  //     point->set_time(time(nullptr));
  //     point->set_x(i);
  //     point->set_y(i);
  //     point->set_z(i);
  //   }
  //   std::ofstream out("out", std::ios::binary | std::ios::out);
  //   if (out) {
  //     std::streampos posBegin = out.tellp();
  //     pointCloud.SerializeToOstream(&out);
  //     std::streampos posEnd = out.tellp();
  //     std::streamoff len = posEnd - posBegin;
  //     printf("output to out file %ld\n", len);
  //   }
  // }

  bool do_test = false;
  auto len = std::filesystem::file_size("out");
  std::vector<uint8_t> buf(len);
  buf.resize(len);
  std::ifstream in("out", std::ios::binary | std::ios::in);
  if (in && in.read((char *)&buf[0], len) && in.gcount() == len) {
    do_test = true;
  }
  long protobufcpp{0};
  long protobufc{0};
  for (int count = 0; count < 1; ++count) {
    std::chrono::time_point<std::chrono::system_clock> tp1 =
        std::chrono::system_clock::now();
    try {
      if (do_test) {
        // printf("input from out file\n");
        test::protobuf::cpp::PointCloud pointCloud;
        pointCloud.ParseFromArray(&buf[0], buf.size());
        // printf("points %d\n", pointCloud.points_size());
        for (size_t i = 0; i < pointCloud.points_size(); ++i) {
          auto result = pointCloud.points(i).x() + pointCloud.points(i).y() +
                        pointCloud.points(i).z();
          // printf("point %f %f %f\n", pointCloud.points(i).x(),
          //        pointCloud.points(i).y(), pointCloud.points(i).z());
        }
      }
    } catch (std::exception const &e) {
      printf("read out error: %s", e.what());
    }
    std::chrono::time_point<std::chrono::system_clock> tp2 =
        std::chrono::system_clock::now();
    Test__Protobuf__C__PointCloud *pointCloud{nullptr};
    try {
      if (do_test) {
        // printf("input from out file\n");
        pointCloud =
            test__protobuf__c__point_cloud__unpack(nullptr, len, &buf[0]);
        if (pointCloud != nullptr) {
          // printf("points %ld\n", pointCloud->n_points);
          for (size_t i = 0; i < pointCloud->n_points; ++i) {
            auto result = pointCloud->points[i]->x + pointCloud->points[i]->y +
                          pointCloud->points[i]->z;
            // printf("point %f %f %f\n", pointCloud->points[i]->x,
            //        pointCloud->points[i]->y, pointCloud->points[i]->z);
          }
          test__protobuf__c__point_cloud__free_unpacked(pointCloud, nullptr);
        } else {
          printf("unpack message error\n");
        }
      }
    } catch (std::exception const &e) {
      printf("read out error: %s", e.what());
    }
    std::chrono::time_point<std::chrono::system_clock> tp3 =
        std::chrono::system_clock::now();
    protobufcpp +=
        (std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1)
             .count());
    protobufc +=
        (std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp2)
             .count());
  }
  printf("protobufcpp %ld\n", protobufcpp);
  printf("protobufc %ld\n", protobufc);

  return 0;
}
