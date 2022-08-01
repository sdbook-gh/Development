#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using tcp = boost::asio::ip::tcp;    // from <boost/asio/ip/tcp.hpp>
namespace http = boost::beast::http; // from <boost/beast/http.hpp>

int main(int argc, char **argv) {
  auto const host = "www.baidu.com"; //要访问的主机名
  auto const port = "80";            // http服务端口

  // The io_context is required for all I/O
  boost::asio::io_context ioc;
  // These objects perform our I/O
  tcp::resolver resolver{ioc};
  tcp::socket socket{ioc};
  // Look up the domain name
  auto const results = resolver.resolve(host, port);
  // Make the connection on the IP address we get from a lookup
  boost::asio::connect(socket, results.begin(), results.end());
  // Set up an HTTP GET request message
  http::request<http::string_body> req{http::verb::get, "/", 11};
  // req.set(http::field::host, host);
  // req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
  //  Send the HTTP request to the remote host
  http::write(socket, req);
  // This buffer is used for reading and must be persisted
  boost::beast::flat_buffer buffer;
  // Declare a container to hold the response
  http::response<http::dynamic_body> http_resp;
  // Receive the HTTP response
  int http_resp_size = http::read(socket, buffer, http_resp);
  std::cout << "http response size " << http_resp_size << std::endl;
  if (http_resp_size > 0) {
    std::fstream out_file("out.txt", std::ios::out | std::ios::binary);
    if (out_file) {
      out_file << http_resp;
      out_file.close();
    }
  }
  // Gracefully close the socket
  boost::system::error_code ec;
  socket.shutdown(tcp::socket::shutdown_both, ec);
  if (ec && ec != boost::system::errc::not_connected) {
    std::cerr << "http close error" << std::endl;
    return -1;
  }
  return 0;
}
