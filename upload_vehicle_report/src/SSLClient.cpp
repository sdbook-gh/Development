#include "SSLClient.h"

#include <cstring>
#include <algorithm>
#include <fstream>
#include <unistd.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <sys/fcntl.h>
#include "ShareResource.h"

using namespace v2x;

SSLClient::SSLClient(const std::string name)
	: clientName(name), server_sock(-1)
{
}

void
SSLClient::setValueToDefault()
{
	ctx = nullptr;
	ssl = nullptr;
	is_initialized = false;
	is_connected = false;
	is_stopped = false;
}
bool
SSLClient::isWorkable()
{
	if (is_stopped)
	{
		return false;
	}
	if (is_connected && is_initialized)
	{
		return true;
	}
	return false;
}

bool
SSLClient::connectToServer()
{
	if (is_stopped)
	{
		return false;
	}
	bool connect_false = false;
	if (!is_connected.compare_exchange_strong(connect_false, true))
	{
		return false;
	}

	std::string ip;
	int port;
	{
		std::unique_lock<std::mutex> lock(ShareResource::configMutex);
		ip = ShareResource::config.getYAMLConfigItem(clientName)["server_ip"].as<std::string>();
		port = ShareResource::config.getYAMLConfigItem(clientName)["server_port"].as<int>();
	}
	struct sockaddr_in servaddr;
	bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(port);
	struct hostent *hostEntry = gethostbyname(ip.c_str());
	if (!hostEntry)
	{
		PRINT_ERROR("get host error: %s(error: %d)", strerror(errno), errno);
		return false;
	}
	memcpy(&servaddr.sin_addr, hostEntry->h_addr_list[0], hostEntry->h_length);
	if ((server_sock = socket(AF_INET, SOCK_STREAM, 0)) == -1)
	{
		PRINT_ERROR("create socket error: %s(error: %d)", strerror(errno), errno);
		return false;
	}
	if (connect(server_sock, (struct sockaddr *)&servaddr, sizeof(servaddr)) == -1)
	{
		PRINT_ERROR("connect socket error: %s(error: %d)", strerror(errno), errno);
		return false;
	}
	PRINT_INFO("connect with server: %s", ip.c_str());
	#ifdef SOCKET_ASYNC
	int fcntlFlag;
	if ((fcntlFlag = fcntl(server_sock, F_GETFL, 0)) < 0)
	{
		PRINT_ERROR("fcntl F_GETFL error: %s(error: %d)", strerror(errno), errno);
		return false;
	}
	fcntlFlag |= O_NONBLOCK;
	if (fcntl(server_sock, F_SETFL, fcntlFlag) < 0)
	{
		PRINT_ERROR("fcntl F_SETFL error: %s(error: %d)", strerror(errno), errno);
		return false;
	}
	PRINT_INFO("change to async mode");
	#endif

	int keepAlive = 1;
	socklen_t optlen = sizeof(keepAlive);
	if (setsockopt(server_sock, SOL_SOCKET, SO_KEEPALIVE, &keepAlive, optlen) < 0)
	{
		PRINT_ERROR("setsockopt SO_KEEPALIVE error: %s(error: %d)", strerror(errno), errno);
		return false;
	}
	return true;
}

bool
SSLClient::init()
{
	if (is_stopped)
	{
		return false;
	}
	bool init_start = false;
	if (!is_initialized.compare_exchange_strong(init_start, true))
	{
		return false;
	}
	if (SSL_library_init() < 0)
	{
		PRINT_SSL_ERROR("SSL_library_init");
		return false;
	}
	if (OpenSSL_add_all_algorithms() < 0)
	{
		PRINT_SSL_ERROR("OpenSSL_add_all_algorithms");
		return false;
	}
	if (SSL_load_error_strings() < 0)
	{
		PRINT_SSL_ERROR("SSL_load_error_strings");
		return false;
	}
	std::string ssl_version;
	std::string ca_cert;
	std::string client_cert;
	std::string client_key;
	{
		std::unique_lock<std::mutex> lock(ShareResource::configMutex);
		ssl_version = ShareResource::config.getYAMLConfigItem(GLOBAL)["ssl_version"].as<std::string>();
		ca_cert = ShareResource::config.getYAMLConfigItem(GLOBAL)["ca_cert"].as<std::string>();
		client_cert = ShareResource::config.getYAMLConfigItem(GLOBAL)["client_cert"].as<std::string>();
		client_key = ShareResource::config.getYAMLConfigItem(GLOBAL)["client_key"].as<std::string>();
	}
	std::transform(ssl_version.begin(), ssl_version.end(), ssl_version.begin(), tolower);
	if (ssl_version == "ssl")
	{
		ctx = SSL_CTX_new(SSLv23_client_method());
		PRINT_INFO("using SSLv23_client_method");
	}
	else if (ssl_version == "tls")
	{
		ctx = SSL_CTX_new(TLS_client_method());
		PRINT_INFO("using TLS_client_method");
	}
	if (ctx == nullptr)
	{
		PRINT_SSL_ERROR("SSL_CTX_new");
		return false;
	}
	if (SSL_CTX_load_verify_locations(ctx, ca_cert.c_str(), 0) <= 0)
	{
		PRINT_SSL_ERROR("SSL_CTX_load_verify_locations");
		return false;
	}
	if (SSL_CTX_use_certificate_file(ctx, client_cert.c_str(), SSL_FILETYPE_PEM) <= 0)
	{
		PRINT_SSL_ERROR("SSL_CTX_use_certificate_file");
		return false;
	}
	if (SSL_CTX_use_PrivateKey_file(ctx, client_key.c_str(), SSL_FILETYPE_PEM) <= 0)
	{
		PRINT_SSL_ERROR("SSL_CTX_use_PrivateKey_file");
		return false;
	}
	if (!SSL_CTX_check_private_key(ctx))
	{
		PRINT_ERROR("Private key does not match public key in certificate");
		return false;
	}
	ssl = SSL_new(ctx);
	return true;
}

bool
SSLClient::handShake()
{
	if (!isWorkable())
	{
		return false;
	}
	if (SSL_set_fd(ssl, server_sock) < 0)
	{
		PRINT_SSL_ERROR("SSL_set_fd");
		return false;
	}
	#ifdef SOCKET_ASYNC
	// set ssl to client mode
	SSL_set_connect_state(ssl);
	int epollfd;
	if ((epollfd = epoll_create(1)) < 0)
	{
		PRINT_ERROR("epoll_create error: %s(error: %d)", strerror(errno), errno);
		return false;
	}
	struct epoll_event epoll_event;
	int events = EPOLLIN | EPOLLOUT | EPOLLERR | EPOLLHUP | EPOLLRDHUP;
	int ret;
	while ((ret = SSL_do_handshake(ssl)) != 1)
	{
		int err = SSL_get_error(ssl, ret);
		if (err == SSL_ERROR_WANT_WRITE)
		{
			events |= EPOLLOUT;
			events &= ~EPOLLIN;
			PRINT_INFO("SSL_ERROR_WANT_WRITE");
		}
		else if (err == SSL_ERROR_WANT_READ)
		{
			events |= EPOLLIN;
			events &= ~EPOLLOUT;
			PRINT_INFO("SSL_ERROR_WANT_READ");
		}
		else
		{
			PRINT_SSL_ERROR("SSL_do_handshake");
		}
		bzero(&epoll_event, sizeof(struct epoll_event));
		epoll_event.events = events;
		if (epoll_ctl(epollfd, EPOLL_CTL_ADD, server_sock, &epoll_event) < 0)
		{
			PRINT_ERROR("epoll_ctl EPOLL_CTL_ADD error: %s(error: %d)", strerror(errno), errno);
			return false;
		}
		struct epoll_event epoll_events[1];
		int epoll_count;
		do
		{
			epoll_count = epoll_wait(epollfd, &epoll_events[0], 1, 1000);
			int bits = EPOLLHUP | EPOLLERR | EPOLLRDHUP;
			if (epoll_count < 0 || (epoll_count > 0 && (epoll_events[0].events & bits) > 0))
			{
				PRINT_ERROR("epoll_wait error: %s(error: %d)", strerror(errno), errno);
				return false;
			}
		}
		while (epoll_count == 0);
		epoll_ctl(epollfd, EPOLL_CTL_DEL, server_sock, &epoll_event);
	}
	close(epollfd);
	#endif
	if (SSL_connect(ssl) < 0)
	{
		PRINT_SSL_ERROR("SSL_connect");
		return false;
	}
	PRINT_INFO("connect with %s encryption", SSL_get_cipher(ssl));
	return true;
}

void
SSLClient::displayServerCertificate()
{
	if (!isWorkable())
	{
		return;
	}
	X509 *cert = SSL_get_peer_certificate(ssl);
	if (cert != NULL)
	{
		PRINT_INFO("Server certificates:");
		char *line;
		line = X509_NAME_oneline(X509_get_subject_name(cert), 0, 0);
		PRINT_INFO("Subject: %s", line);
		free(line);
		line = X509_NAME_oneline(X509_get_issuer_name(cert), 0, 0);
		PRINT_INFO("Issuer: %s", line);
		free(line);
		X509_free(cert);
	}
	else
	{
		PRINT_INFO("No certificates.");
	}
}

bool
SSLClient::closeConnection()
{
	bool connect_true = true;
	if (!is_connected.compare_exchange_strong(connect_true, false))
	{
		return false;
	}
	std::unique_lock<std::mutex> sendLock(send_mutex);
	std::unique_lock<std::mutex> receiveLock(receive_mutex);
	#ifdef SOCKET_ASYNC
	if (send_epoll_fd > 0)
	{
		epoll_ctl(send_epoll_fd, EPOLL_CTL_DEL, server_sock, &send_epoll_event);
		close(send_epoll_fd);
		send_epoll_fd = -1;
	}
	if (receive_epoll_fd > 0)
	{
		epoll_ctl(receive_epoll_fd, EPOLL_CTL_DEL, server_sock, &receive_epoll_event);
		close(receive_epoll_fd);
		receive_epoll_fd = -1;
	}
	#endif
	if (ssl)
	{
		SSL_shutdown(ssl);
		SSL_free(ssl);
		ssl = nullptr;
	}
	if (ctx)
	{
		SSL_CTX_free(ctx);
		ctx = nullptr;
	}
	if (server_sock != -1)
	{
		::close(server_sock);
		server_sock = -1;
		PRINT_INFO("closeConnection with server");
	}
	return true;
}

#define CHECK_WORKABLE(info_str) \
if (!isWorkable()) \
{ \
PRINT_ERROR(info_str); \
return ERROR_STATUS_STOPPED; \
}

#define CHECK_TIMEOUT_START(timeout_second) \
std::chrono::time_point<std::chrono::system_clock> __start; \
if (timeout_second > 0) \
{ \
__start = std::chrono::system_clock::now(); \
}

#define CHECK_TIMEOUT_FINISH(timeout_second, info_str) \
if (timeout_second > 0) \
{ \
std::chrono::time_point<std::chrono::system_clock> __current = std::chrono::system_clock::now(); \
int __seconds = std::chrono::duration_cast<std::chrono::seconds>(__current - __start).count(); \
if (__seconds >= timeout_second) \
{ \
PRINT_ERROR(info_str); \
return ERROR_NETWORK_TIMEOUT; \
} \
}

int
SSLClient::send(const uint8_t *const buffer, const size_t offset, const size_t bufferLength, const size_t expectLength, int timeout_second)
{
	std::unique_lock<std::mutex> sendLock(send_mutex);

	CHECK_WORKABLE("send error: not workable")

	#ifdef SOCKET_ASYNC
	CHECK_TIMEOUT_START(timeout_second)
	if (send_epoll_fd < 0)
	{
		send_epoll_fd = epoll_create(1);
		if (send_epoll_fd < 0)
		{
			PRINT_ERROR("epoll_create error: %s(error: %d)", strerror(errno), errno);
			return ERROR_NETWORK_EPOLL;
		}
		bzero(&send_epoll_event, sizeof(send_epoll_event));
		send_epoll_event.events = EPOLLOUT | EPOLLERR | EPOLLHUP | EPOLLRDHUP;
		if (epoll_ctl(send_epoll_fd, EPOLL_CTL_ADD, server_sock, &send_epoll_event) < 0)
		{
			PRINT_ERROR("epoll_ctl EPOLL_CTL_ADD error: %s(error: %d)", strerror(errno), errno);
			return ERROR_NETWORK_EPOLL;
		}
	}
	#endif

	int send_size = 0;
	size_t total_send_size = 0;
	do
	{
		#ifdef SOCKET_ASYNC
		CHECK_TIMEOUT_FINISH(timeout_second, "send error: timeout")
		#endif
		CHECK_WORKABLE("send error: not workable")

		#ifdef SOCKET_ASYNC
		struct epoll_event epoll_events[1];
		int epoll_count = epoll_wait(send_epoll_fd, &epoll_events[0], 1, 1000);
		int bits = EPOLLHUP | EPOLLERR | EPOLLRDHUP;
		if (epoll_count < 0 || (epoll_count > 0 && (epoll_events[0].events & bits) > 0))
		{
			PRINT_ERROR("epoll_wait error: %s(error: %d)", strerror(errno), errno);
			return ERROR_NETWORK_EPOLL;
		}
		#endif
		int sock_error;
		socklen_t len = sizeof(sock_error);
		int ret = getsockopt(server_sock, SOL_SOCKET, SO_ERROR, &sock_error, &len);
		if (ret != 0)
		{
			PRINT_ERROR("send getsockopt SO_ERROR error: %s(error: %d)", strerror(ret), ret);
			return ERROR_NETWORK;
		}
		if (sock_error != 0)
		{
			PRINT_ERROR("send error: %s(error: %d)", strerror(sock_error), sock_error);
			return ERROR_NETWORK;
		}
		send_size = SSL_write(ssl, buffer + offset + total_send_size, bufferLength - total_send_size);
		if (send_size <= 0)
		{
			if (errno == EAGAIN || errno == EINTR || SSL_get_error(ssl, send_size) == SSL_ERROR_WANT_WRITE)
			{
//				PRINT_INFO("send need to try again");
				continue;
			}
			PRINT_SSL_ERROR("SSL_write");
			return ERROR_NETWORK;
		}
		total_send_size += send_size;
		if (expectLength <= 0 || timeout_second <= 0)
		{
			break;
		}
	}
	while (total_send_size < expectLength);
	return total_send_size;
}

int
SSLClient::receive(uint8_t *const buffer, const size_t offset, const size_t bufferLength, const size_t expectLength, int timeout_second)
{
	std::unique_lock<std::mutex> receiveLock(receive_mutex);
	CHECK_WORKABLE("receive error: not workable")

	#ifdef SOCKET_ASYNC
	CHECK_TIMEOUT_START(timeout_second)
	if (receive_epoll_fd < 0)
	{
		receive_epoll_fd = epoll_create(1);
		if (receive_epoll_fd < 0)
		{
			PRINT_ERROR("epoll_create error: %s(error: %d)", strerror(errno), errno);
			return ERROR_NETWORK_EPOLL;
		}
		bzero(&receive_epoll_event, sizeof(receive_epoll_event));
		receive_epoll_event.events = EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLRDHUP;
		if (epoll_ctl(receive_epoll_fd, EPOLL_CTL_ADD, server_sock, &receive_epoll_event) < 0)
		{
			PRINT_ERROR("epoll_ctl EPOLL_CTL_ADD error: %s(error: %d)", strerror(errno), errno);
			return ERROR_NETWORK_EPOLL;
		}
	}
	#endif

	int receive_size = 0;
	size_t total_receive_size = 0;
	do
	{
		#ifdef SOCKET_ASYNC
		CHECK_TIMEOUT_FINISH(timeout_second, "receive error: timeout")
		#endif
		CHECK_WORKABLE("receive error: not workable")

		#ifdef SOCKET_ASYNC
		struct epoll_event epoll_events[1];
		int epoll_count = epoll_wait(receive_epoll_fd, &epoll_events[0], 1, 1000);
		int bits = EPOLLHUP | EPOLLERR | EPOLLRDHUP;
		if (epoll_count < 0 || (epoll_count > 0 && (epoll_events[0].events & bits) > 0))
		{
			PRINT_ERROR("epoll_wait error: %s(error: %d)", strerror(errno), errno);
			return ERROR_NETWORK_EPOLL;
		}
		#endif
		int sock_error;
		socklen_t len = sizeof(sock_error);
		int ret = getsockopt(server_sock, SOL_SOCKET, SO_ERROR, &sock_error, &len);
		if (ret != 0)
		{
			PRINT_ERROR("receive getsockopt SO_ERROR error: %s(error: %d)", strerror(ret), ret);
			return ERROR_NETWORK;
		}
		if (sock_error != 0)
		{
			PRINT_ERROR("receive error: %s(error: %d)", strerror(sock_error), sock_error);
			return ERROR_NETWORK;
		}

		receive_size = SSL_read(ssl, buffer + offset + total_receive_size, bufferLength - total_receive_size);
		if (receive_size <= 0)
		{
			if (errno == EAGAIN || errno == EINTR || SSL_get_error(ssl, receive_size) == SSL_ERROR_WANT_READ)
			{
//				PRINT_INFO("receive need to try again");
				continue;
			}
			PRINT_SSL_ERROR("SSL_read");
			return ERROR_NETWORK;
		}
		total_receive_size += receive_size;
		if (expectLength <= 0 || timeout_second < 0)
		{
			break;
		}
	}
	while (total_receive_size < expectLength);
	return total_receive_size;
}

void
SSLClient::stopWork()
{
	bool stop_false = false;
	if (!is_stopped.compare_exchange_strong(stop_false, true))
	{
		return;
	}
	if (!isWorkable())
	{
		return;
	}
	closeConnection();
}

void
SSLClient::reset()
{
	stopWork();
	setValueToDefault();
}
