#include "amessage.pb.h"
#include "amessage.v2.pb-c.h"
#include "amessage_generated.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "iguana/pb_reader.hpp"
#include "iguana/pb_writer.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

static bool print_log{false};

class ScopedTimer {
public:
  ScopedTimer(const char *name)
      : m_name(name), m_beg(std::chrono::high_resolution_clock::now()) {}
  ScopedTimer(const char *name, uint64_t &ns) : ScopedTimer(name) {
    m_ns = &ns;
  }
  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
    if (m_ns)
      *m_ns = dur.count();
    else
      std::cout << std::left << std::setw(45) << m_name << " : " << std::right
                << std::setw(12) << dur.count() << " us\n";
  }

private:
  const char *m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
  uint64_t *m_ns = nullptr;
};

void pbc_serialize(std::string &out_buf) {
  thread_local Test__Protobuf__C__PointCloud pointCloud;
  test__protobuf__c__point_cloud__init(&pointCloud);
  pointCloud.height = 1080;
  pointCloud.has_height = 1;
  pointCloud.width = 1920;
  pointCloud.has_width = 1;
  pointCloud.point_step = 1;
  pointCloud.has_point_step = 1;
  pointCloud.row_step = 1;
  pointCloud.has_row_step = 1;
  thread_local std::vector<Test__Protobuf__C__PointField> point_vec(
      pointCloud.height * pointCloud.width);
  point_vec.resize(pointCloud.height * pointCloud.width);
  thread_local std::vector<Test__Protobuf__C__PointField *> point_ptr_vec(
      pointCloud.height * pointCloud.width);
  point_ptr_vec.resize(pointCloud.height * pointCloud.width);
  for (int i = 0; i < pointCloud.height * pointCloud.width; ++i) {
    Test__Protobuf__C__PointField *point = &point_vec[i];
    test__protobuf__c__point_field__init(point);
    point->time = time(nullptr);
    point->has_time = 1;
    point->x = i;
    point->has_x = 1;
    point->y = i;
    point->has_y = 1;
    point->z = i;
    point->has_z = 1;
    point_ptr_vec[i] = point;
  }
  pointCloud.points = &point_ptr_vec[0];
  pointCloud.n_points = point_ptr_vec.size();
  auto len = test__protobuf__c__point_cloud__get_packed_size(&pointCloud);
  out_buf.resize(len);
  test__protobuf__c__point_cloud__pack(&pointCloud, (uint8_t *)out_buf.data());
}

void pbc_deserialize(std::string &in_buf) {
  Test__Protobuf__C__PointCloud *pointCloud =
      test__protobuf__c__point_cloud__unpack(nullptr, in_buf.size(),
                                             (uint8_t *)in_buf.data());
  if (pointCloud != nullptr) {
    if (print_log)
      printf("%s points %ld\n", __FUNCTION__, pointCloud->n_points);
    double result = 0;
    for (size_t i = 0; i < pointCloud->n_points; ++i) {
      result += (pointCloud->points[i]->x + pointCloud->points[i]->y +
                 pointCloud->points[i]->z);
    }
    test__protobuf__c__point_cloud__free_unpacked(pointCloud, nullptr);
    if (print_log)
      printf("%s %f\n", __FUNCTION__, result);
  } else {
    printf("unpack message error\n");
  }
}

void pb_serialize(std::string &out_buf) {
  test::protobuf::cpp::PointCloud pointCloud;
  pointCloud.set_height(1080);
  pointCloud.set_width(1920);
  pointCloud.set_point_step(1);
  pointCloud.set_row_step(1);
  for (int i = 0; i < pointCloud.height() * pointCloud.width(); i++) {
    auto *point = pointCloud.add_points();
    point->set_time(time(nullptr));
    point->set_x(i);
    point->set_y(i);
    point->set_z(i);
  }
  pointCloud.SerializeToString(&out_buf);
}

void pb_deserialize(std::string &out_buf) {
  test::protobuf::cpp::PointCloud pointCloud;
  pointCloud.ParseFromString(out_buf);
  if (print_log)
    printf("%s points %d\n", __FUNCTION__, pointCloud.points_size());
  double result = 0;
  for (size_t i = 0; i < pointCloud.points_size(); ++i) {
    result += (pointCloud.points(i).x() + pointCloud.points(i).y() +
               pointCloud.points(i).z());
  }
  if (print_log)
    printf("%s %f\n", __FUNCTION__, result);
}

namespace pbs {
struct PointField : iguana::pb_base {
  uint32_t time;
  double x;
  double y;
  double z;
  double distance;
  double pitch;
  double yaw;
  uint32_t intensity;
  uint32_t ring;
};
REFLECTION(PointField, time, x, y, z, distance, pitch, yaw, intensity, ring);

struct PointCloud : iguana::pb_base {
  uint32_t height; // 点云二维结构高度。
  uint32_t width; // 点云二维结构宽度，点云数量有效范围由具体传感器决定。
  uint32_t point_step; // 一个点云的长度。单位：字节
  uint32_t row_step;   // 一行点云的长度。单位：字节
  std::vector<PointField> points;
};
REFLECTION(PointCloud, height, width, point_step, row_step, points);

} // namespace pbs

void pbs_serialize(std::string &out_buf) {
  pbs::PointCloud pointCloud;
  pointCloud.width = 1080;
  pointCloud.height = 1920;
  pointCloud.point_step = 1;
  pointCloud.row_step = 1;
  pointCloud.points.resize(pointCloud.width * pointCloud.height);
  for (int i = 0; i < pointCloud.width * pointCloud.height; ++i) {
    auto *point = &pointCloud.points[i];
    point->time = time(nullptr);
    point->x = i;
    point->y = i;
    point->z = i;
  }
  iguana::to_pb(pointCloud, out_buf);
}

void pbs_deserialize(std::string &out_buf) {
  pbs::PointCloud pointCloud;
  iguana::from_pb(pointCloud, out_buf);
  if (print_log)
    printf("%s points %ld\n", __FUNCTION__, pointCloud.points.size());
  double result = 0;
  for (size_t i = 0; i < pointCloud.points.size(); ++i) {
    result += (pointCloud.points[i].x + pointCloud.points[i].y +
               pointCloud.points[i].z);
  }
  if (print_log)
    printf("%s %f\n", __FUNCTION__, result);
}

void fb_serialize(std::string &out_buf) {
  // using namespace MyGame::Sample;
  // std::vector<flatbuffers::Offset<Weapon>> weapons_vector;
  // weapons_vector.push_back([&] {
  //   auto name = builder.CreateString("Sword");
  //   WeaponBuilder wb(builder);
  //   wb.add_name(name);
  //   return wb.Finish();
  // }());
  // weapons_vector.push_back([&] {
  //   auto name = builder.CreateString("Axe");
  //   WeaponBuilder wb(builder);
  //   wb.add_name(name);
  //   return wb.Finish();
  // }());
  // auto weapons = builder.CreateVector(weapons_vector);
  // auto position = Vec3(1.0f, 2.0f, 3.0f);
  // auto name = builder.CreateString("MyMonster");
  // unsigned char inv_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  // auto inventory = builder.CreateVector(inv_data, 10);
  // std::vector<flatbuffers::Offset<PointField>> pointsVec;
  // for (int i = 0; i < 10; ++i) {
  //   PointFieldBuilder pointFieldBuilder(builder);
  //   pointFieldBuilder.add_time(time(nullptr));
  //   pointsVec.emplace_back(pointFieldBuilder.Finish());
  // }
  // auto points = builder.CreateVector(pointsVec);
  // MonsterBuilder monsterBuilder(builder);
  // monsterBuilder.add_pos(&position);
  // monsterBuilder.add_mana(150);
  // monsterBuilder.add_hp(80);
  // monsterBuilder.add_name(name);
  // monsterBuilder.add_inventory(inventory);
  // monsterBuilder.add_color(Color::Color_Red);
  // monsterBuilder.add_weapons(weapons);
  // monsterBuilder.add_equipped_type(Equipment::Equipment_Weapon);
  // monsterBuilder.add_equipped(weapons_vector[0].Union());
  // monsterBuilder.add_points(points);
  // auto orc = monsterBuilder.Finish();
  // builder.Finish(orc);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<test::flatbuffers::cpp::PointField>>
      pointsVec;
  for (int i = 0; i < 1920 * 1080; ++i) {
    test::flatbuffers::cpp::PointFieldBuilder pointFieldBuilder(builder);
    pointFieldBuilder.add_time(time(nullptr));
    pointFieldBuilder.add_x(i);
    pointFieldBuilder.add_y(i);
    pointFieldBuilder.add_z(i);
    auto pointField = pointFieldBuilder.Finish();
    pointsVec.push_back(pointField);
  }
  auto points = builder.CreateVector(pointsVec);
  test::flatbuffers::cpp::PointCloudBuilder pointCloudBuilder(builder);
  pointCloudBuilder.add_width(1080);
  pointCloudBuilder.add_height(1920);
  pointCloudBuilder.add_point_step(1);
  pointCloudBuilder.add_row_step(1);
  pointCloudBuilder.add_points(points);
  builder.Finish(pointCloudBuilder.Finish());
  std::string serial_str(builder.GetBufferPointer(),
                         builder.GetBufferPointer() + builder.GetSize());
  out_buf.swap(serial_str);
}

void fb_deserialize(std::string &out_buf) {
  // using namespace MyGame::Sample;
  // auto monster = GetMonster(builder.GetBufferPointer());
  // if (monster->points() != nullptr) {
  //   printf("%d\n", monster->points()->size());
  // }

  auto pointCloud = test::flatbuffers::cpp::GetPointCloud(out_buf.data());
  if (pointCloud == nullptr) {
    printf("pointCloud null\n");
    return;
  }
  if (pointCloud->points() == nullptr) {
    printf("points null\n");
    return;
  }
  if (print_log)
    printf("%s points %d\n", __FUNCTION__, pointCloud->points()->size());

  double result = 0;
  for (auto i = 0u; i < pointCloud->points()->size(); ++i) {
    auto point = pointCloud->points()->Get(i);
    auto exist =
        flatbuffers::IsFieldPresent(point,
                                    test::flatbuffers::cpp::PointField::VT_X) &&
        flatbuffers::IsFieldPresent(point,
                                    test::flatbuffers::cpp::PointField::VT_Y) &&
        flatbuffers::IsFieldPresent(point,
                                    test::flatbuffers::cpp::PointField::VT_Z);
    if (exist) {
      result += (point->x() + point->y() + point->z());
    }
  }
  if (print_log)
    printf("%s %f\n", __FUNCTION__, result);
}

void bench4(int count) {
  {
    ScopedTimer pbc_serialize_timer("pbc_serialize");
    for (int i = 0; i < count; ++i) {
      std::string pbc_str;
      pbc_serialize(pbc_str);
    }
  }
  {
    ScopedTimer pb_serialize_timer("pb_serialize");
    for (int i = 0; i < count; ++i) {
      std::string pb_str;
      pb_serialize(pb_str);
    }
  }
  {
    ScopedTimer pbs_serialize_timer("pbs_serialize");
    for (int i = 0; i < count; ++i) {
      std::string pbs_str;
      pbs_serialize(pbs_str);
    }
  }
  {
    ScopedTimer fb_serialize_timer("fb_serialize");
    for (int i = 0; i < count; ++i) {
      // flatbuffers::FlatBufferBuilder fb_builder;
      // fb_serialize(fb_builder);
      std::string fb_str;
      fb_serialize(fb_str);
    }
  }
  {
    std::string pbc_str;
    pbc_serialize(pbc_str);
    ScopedTimer pbc_deserialize_timer("pbc_deserialize");
    for (int i = 0; i < count; ++i) {
      pbc_deserialize(pbc_str);
    }
  }
  {
    std::string pb_str;
    pb_serialize(pb_str);
    ScopedTimer pb_deserialize_timer("pb_deserialize");
    for (int i = 0; i < count; ++i) {
      pb_deserialize(pb_str);
    }
  }
  {
    std::string pbs_str;
    pbs_serialize(pbs_str);
    ScopedTimer pbs_deserialize_timer("pbs_deserialize");
    for (int i = 0; i < count; ++i) {
      pbs_deserialize(pbs_str);
    }
  }
  {
    // flatbuffers::FlatBufferBuilder fb_builder;
    // fb_serialize(fb_builder);
    std::string fb_str;
    fb_serialize(fb_str);
    ScopedTimer fb_deserialize_timer("fb_deserialize");
    for (int i = 0; i < count; ++i) {
      // fb_deserialize(fb_builder);
      fb_deserialize(fb_str);
    }
  }

  // std::string pbs_str;
  // pbs_serialize(pbs_str);
  // pb_deserialize(pbs_str);
  // pbc_deserialize(pbs_str);
  // std::string pb_str;
  // pb_serialize(pb_str);
  // pbs_deserialize(pbs_str);
  // pbc_deserialize(pbs_str);
  // std::string pbc_str;
  // pbc_serialize(pbc_str);
  // pbs_deserialize(pbc_str);
  // pb_deserialize(pbc_str);

  // flatbuffers::FlatBufferBuilder fb_str;
  // fb_serialize(fb_str);
  // fb_deserialize(fb_str);
}

int main() {
  // bench(100000);
  // std::cout << "----------------------------------------\n";
  // bench2(100000);
  // std::cout << "----------------------------------------\n";
  // bench3(100000);
  bench4(100);
  return 0;
}
