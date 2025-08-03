#include <arpa/inet.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <sys/socket.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "image.pb.h"

int main() {
  // 1. 建立 TCP 监听
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(12345);
  addr.sin_addr.s_addr = INADDR_ANY;
  bind(listen_fd, (sockaddr*)&addr, sizeof(addr));
  listen(listen_fd, 1);

  // 2. 接受连接并读取数据到缓冲区
  int conn = accept(listen_fd, nullptr, nullptr);
  std::vector<uint8_t> recv_buf(10 * 1024 * 1024); // 预分配 10MB

  size_t total_received = 0;
  ssize_t n = 0;
  while (total_received < recv_buf.size()) {
    n = recv(conn, recv_buf.data() + total_received, recv_buf.size() - total_received, 0);
    if (n < 0) {
      perror("recv");
      return -1;
    } else if (n == 0) {
      // Connection closed
      break;
    }
    total_received += n;
  }
  if (total_received == 0) {
    std::cerr << "No data received\n";
    return -1;
  }
  std::cout << total_received << "\n";
  recv_buf.resize(total_received);

  // 3. 使用 ArrayInputStream 零拷贝反序列化
  google::protobuf::io::ArrayInputStream ais(recv_buf.data(), recv_buf.size());
  ImageData msg;
  if (!msg.ParseFromZeroCopyStream(&ais)) {
    std::cerr << "Parse failed\n";
    return -1;
  }

  // 4. 将图像保存到文件（或进一步处理）
  std::ofstream out("recv_image.jpg", std::ios::binary);
  const auto& data = msg.data(); // data() 返回 const std::string&
  out.write(data.data(), data.size());
  close(conn);
  close(listen_fd);
  return 0;
}
