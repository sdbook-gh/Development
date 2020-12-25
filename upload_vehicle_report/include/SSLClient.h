#ifndef SSL_CLIENT
#define SSL_CLIENT

//#define SOCKET_ASYNC

#include <string>
#include <atomic>
#include <mutex>
#include <cinttypes>
#include <cstdarg>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <sys/epoll.h>

namespace v2x
{
class SSLClient
{
public:
	SSLClient(const std::string name);

	virtual ~SSLClient()
	{
		closeConnection();
	}

	bool
	connectToServer();

	bool
	init();

	bool
	handShake();

	void
	displayServerCertificate();

	bool
	closeConnection();

	int
	receive(uint8_t *const buffer, const size_t offset, const size_t bufferLength, const size_t expectLength, int timeout_second = 0);

	int
	send(const uint8_t *const buffer, const size_t offset, const size_t bufferLength, const size_t expectLength, int timeout_second = 0);

	bool
	isWorkable();

	void
	stopWork();

	void
	reset();

	const std::string
	getName() const
	{
		return clientName;
	}

private:
	void
	setValueToDefault();
	std::string clientName;
	int server_sock;
	SSL_CTX *ctx = nullptr;
	SSL *ssl = nullptr;
	std::atomic<bool> is_initialized{false};
	std::atomic<bool> is_connected{false};
	std::mutex send_mutex;
	std::mutex receive_mutex;
	std::atomic<bool> is_stopped{false};
	#ifdef SOCKET_ASYNC
	int send_epoll_fd = -1;
	struct epoll_event send_epoll_event;
	int receive_epoll_fd = -1;
	struct epoll_event receive_epoll_event;
	#endif
};
} // namespace v2x
#endif
