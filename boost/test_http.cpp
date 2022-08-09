#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace boost::asio::ip;    // from <boost/asio/ip/tcp.hpp>
using namespace boost::beast; // from <boost/beast/http.hpp>

int sync_process_main(int argc, char **argv) {
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

// Report a failure
void fail(boost::system::error_code ec, char const * const what) {
  std::cerr << what << ": " << ec.message() << "\n";
}

// Performs an HTTP GET and prints the response
class session : public std::enable_shared_from_this<session> {
private:
  tcp::resolver resolver_;
  tcp::socket socket_;
  boost::beast::flat_buffer buffer_; // (Must persist between reads)
  http::request<http::empty_body> req_;
  http::response<http::string_body> res_;
public:
  // Resolver and socket require an io_context
  explicit session(boost::asio::io_context &ioc) : resolver_(ioc), socket_(ioc) {}
  // Start the asynchronous operation
  void run(char const *host, char const *port, char const *target, int version) {
    // Set up an HTTP GET request message
    req_.version(version);
    req_.method(http::verb::get);
    req_.target(target);
    req_.set(http::field::host, host);
    req_.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    // Look up the domain name
    resolver_.async_resolve(host, port, std::bind(&session::on_resolve, shared_from_this(), std::placeholders::_1, std::placeholders::_2));
  }
  void on_resolve(boost::system::error_code ec, tcp::resolver::results_type results) {
    if (ec) {
      return fail(ec, "resolve");
    }
    // Make the connection on the IP address we get from a lookup
    boost::asio::async_connect(socket_, results.begin(), results.end(), std::bind(&session::on_connect, shared_from_this(), std::placeholders::_1));
  }
  void on_connect(boost::system::error_code ec) {
    if (ec) {
      return fail(ec, "connect");
    }
    // Send the HTTP request to the remote host
    http::async_write(socket_, req_, std::bind(&session::on_write, shared_from_this(), std::placeholders::_1, std::placeholders::_2));
  }
  void on_write(boost::system::error_code ec, std::size_t bytes_transferred) {
    boost::ignore_unused(bytes_transferred);
    if (ec) {
      return fail(ec, "write");
    }
    // Receive the HTTP response
    http::async_read(socket_, buffer_, res_, std::bind(&session::on_read, shared_from_this(), std::placeholders::_1, std::placeholders::_2));
  }
  void on_read(boost::system::error_code ec, std::size_t bytes_transferred) {
    boost::ignore_unused(bytes_transferred);
    if (ec) {
      return fail(ec, "read");
    }
    // Write the message to standard out
    std::cout << res_ << std::endl;
    // Gracefully close the socket
    socket_.shutdown(tcp::socket::shutdown_both, ec);
    // not_connected happens sometimes so don't bother reporting it.
    if (ec && ec != boost::system::errc::not_connected) {
      return fail(ec, "shutdown");
    }
    // If we get here then the connection is closed gracefully
  }
};
int async_process_main(int argc, char **argv) {
  auto const host = "www.baidu.com";   //要访问的主机名
  auto const port = "80";              // http服务端口
  auto const target = "/"; //要获取的文档
  int version = 11;
  // The io_context is required for all I/O
  boost::asio::io_context ioc;
  // Launch the asynchronous operation
  std::make_shared<session>(ioc)->run(host, port, target, version);
  // Run the I/O service. The call will return when
  // the get operation is complete.
  ioc.run();

  return EXIT_SUCCESS;
}

#include <boost/asio/spawn.hpp>
// Performs an HTTP GET and prints the response
void do_session(std::string const &host, std::string const &port, std::string const &target, int version, boost::asio::io_context &ioc, boost::asio::yield_context yield) {
  boost::system::error_code ec;
  // These objects perform our I/O
  tcp::resolver resolver{ioc};
  tcp::socket socket{ioc};
  // Look up the domain name
  auto const results = resolver.async_resolve(host, port, yield[ec]);
  if (ec) {
    return fail(ec, "resolve");
  }
  // Make the connection on the IP address we get from a lookup
  boost::asio::async_connect(socket, results.begin(), results.end(), yield[ec]);
  if (ec) {
    return fail(ec, "connect");
  }
  // Set up an HTTP GET request message
  http::request<http::string_body> req{http::verb::get, target, version};
  req.set(http::field::host, host);
  req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
  // Send the HTTP request to the remote host
  http::async_write(socket, req, yield[ec]);
  if (ec) {
    return fail(ec, "write");
  }
  // This buffer is used for reading and must be persisted
  boost::beast::flat_buffer buffer;
  // Declare a container to hold the response
  http::response<http::dynamic_body> res;
  // Receive the HTTP response
  http::async_read(socket, buffer, res, yield[ec]);
  if (ec) {
    return fail(ec, "read");
  }
  // Write the message to standard out
  std::cout << res << std::endl;
  // Gracefully close the socket
  socket.shutdown(tcp::socket::shutdown_both, ec);
  // not_connected happens sometimes
  // so don't bother reporting it.
  //
  if (ec && ec != boost::system::errc::not_connected) {
    return fail(ec, "shutdown");
  }
  // If we get here then the connection is closed gracefully
}
int coroutine_process_main(int argc, char **argv) {
  auto const host = "www.baidu.com";   //要访问的主机名
  auto const port = "80";              // http服务端口
  auto const target = "/"; //要获取的文档
  int version = 11;
  // The io_context is required for all I/O
  boost::asio::io_context ioc;
  // Launch the asynchronous operation
  boost::asio::spawn(ioc, std::bind(&do_session, std::string(host), std::string(port), std::string(target), version, std::ref(ioc), std::placeholders::_1));
  // Run the I/O service. The call will return when
  // the get operation is complete.
  ioc.run();
  return EXIT_SUCCESS;
}
