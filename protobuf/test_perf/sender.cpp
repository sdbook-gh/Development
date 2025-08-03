#include <arpa/inet.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <sys/socket.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "image.pb.h"

int main() {
  // 1. 读取图像到缓冲区
  std::ifstream in("image.jpg", std::ios::binary | std::ios::ate);
  if (!in) {
    perror("open");
    return -1;
  }
  std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(size);
  if (!in.read(reinterpret_cast<char*>(buf.data()), size)) {
    perror("read");
    return -1;
  }

  // 2. 构造 Protobuf 消息并将 data 指向已有缓冲区
  ImageData msg;
  // 直接将字节数组指向 msg.data，无额外拷贝
  msg.set_data(buf.data(), size);

  // 3. 为序列化分配同等大小的输出区，并创建 ArrayOutputStream
  std::vector<uint8_t> out_buf(msg.ByteSizeLong());
  google::protobuf::io::ArrayOutputStream aos(out_buf.data(), out_buf.size());

  // 零拷贝序列化
  if (!msg.SerializeToZeroCopyStream(&aos)) {
    std::cerr << "Serialize failed\n";
    return -1;
  }

  // 4. 建立 TCP 连接并发送序列化数据
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in srv{};
  srv.sin_family = AF_INET;
  srv.sin_port = htons(12345);
  inet_pton(AF_INET, "127.0.0.1", &srv.sin_addr);
  if (connect(sock, (sockaddr*)&srv, sizeof(srv)) < 0) {
    perror("connect");
    return -1;
  }
  size_t total = out_buf.size();
  size_t sent = 0;
  while (sent < total) {
    ssize_t ret = send(sock, out_buf.data() + sent, total - sent, 0);
    if (ret < 0) {
      perror("send");
      close(sock);
      return -1;
    }
    sent += ret;
  }
  std::cout << sent << "\n";
  close(sock);
  return 0;
}
