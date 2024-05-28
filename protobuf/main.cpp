#include "amessage.pb-c.h"
#include "amessage.pb.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, const char *argv[]) {
  {
    Test__Protobuf__C__AMessage msg = TEST__PROTOBUF__C__AMESSAGE__INIT;
    msg.a = 1;
    msg.has_b = 1;
    msg.b = 2;
    msg.n_c = 10;
    std::vector<int32_t> buf_c(10);
    for (int i = 0; i < 10; ++i) {
      buf_c.emplace_back(i);
    }
    msg.c = &buf_c[0];
    msg.n_d = 10;
    std::vector<char *> buf_d(msg.n_d);
    buf_d.resize(msg.n_d);
    std::vector<std::vector<char>> val_array_d(msg.n_d);
    val_array_d.resize(msg.n_d);
    for (int i = 0; i < msg.n_d; i++) {
      auto str = "string " + std::to_string(i);
      val_array_d[i].resize(str.size() + 1);
      strcpy(&val_array_d[i][0], str.c_str());
      buf_d[i] = &val_array_d[i][0];
    }
    msg.d = &buf_d[0];
    std::vector<uint8_t> buf_e(10);
    for (int i = 0; i < 10; ++i) {
      buf_e.emplace_back(i);
    }
    msg.e.len = 10;
    msg.e.data = &buf_e[0];
    msg.f = nullptr;
    msg.n_g = 10;
    std::vector<Test__Protobuf__C__Submessage2> val_array_g(msg.n_g);
    val_array_g.resize(msg.n_g);
    std::vector<Test__Protobuf__C__Submessage2 *> buf_g(msg.n_g);
    buf_g.resize(msg.n_g);
    for (int i = 0; i < msg.n_g; ++i) {
      test__protobuf__c__submessage2__init(&val_array_g[i]);
      val_array_g[i].value = i;
      buf_g[i] = &val_array_g[i];
    }
    msg.g = &buf_g[0];

    auto len = test__protobuf__c__amessage__get_packed_size(&msg);
    std::vector<uint8_t> buf(len);
    buf.resize(len);
    test__protobuf__c__amessage__pack(&msg, &buf[0]);
    printf("Writing %ld serialized bytes\n", len);
    std::ofstream out("out", std::ios::binary | std::ios::out);
    if (out) {
      out.write((char const *)&buf[0], len);
      printf("output to out file\n");
    }
  }
  {
    Test__Protobuf__C__AMessage *msg{nullptr};
    try {
      auto len = std::filesystem::file_size("out");
      std::vector<uint8_t> buf(len);
      buf.resize(len);
      std::ifstream in("out", std::ios::binary | std::ios::in);
      if (in && in.read((char *)&buf[0], len) && in.gcount() == len) {
        printf("input from out file\n");
        msg = test__protobuf__c__amessage__unpack(nullptr, len, &buf[0]);
        if (msg != nullptr) {
          printf("a=%d\n", msg->a);
          if (msg->has_b != 0)
            printf("b=%d\n", msg->b);
          test__protobuf__c__amessage__free_unpacked(msg, nullptr);
        } else {
          printf("unpack message error\n");
        }
        test::protobuf::cpp::AMessage amsg;
        amsg.ParseFromArray(&buf[0], buf.size());
        printf("a=%d\n", amsg.a());
        if (amsg.has_b())
          printf("b=%d\n", amsg.b());
        for (int i = 0; i < amsg.d_size(); ++i) {
          printf("d=%s\n", amsg.d().at(i).c_str());
        }
        if (amsg.has_f()) {
          printf("f=%d\n", amsg.f().value());
        }
        for (int i = 0; i < amsg.g_size(); ++i) {
          printf("g=%d\n", amsg.g().at(i).value());
        }
      }
    } catch (std::exception const &e) {
      printf("read out error: %s", e.what());
    }
  }
  return 0;
}
