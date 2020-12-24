#ifndef CONCURRENT_DEQUE_H
#define CONCURRENT_DEQUE_H

#include <vector>
#include <queue>
#include <deque>
#include <list>
#include <mutex>
#include <atomic>

template<typename T>
class AtomicVector
{
#define GET_LOCK \
    while (_flag.test_and_set()); \
    std::shared_ptr<int> guard(nullptr, [&](void *){_flag.clear();});

public:
	explicit AtomicVector()
	{
		_flag.clear();
	}

	void
	push(const T &value)
	{
		GET_LOCK
		_collection.push_back(value);
	}

	void
	clear(void)
	{
		GET_LOCK
		_collection.clear();
	}

	int
	size()
	{
		GET_LOCK
		return _collection.size();
	}

	bool
	empty()
	{
		GET_LOCK
		return _collection.empty();
	}

	bool
	pop()
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		_collection.pop_back();
		return true;
	}

	bool
	pop(T &value) noexcept
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		value = _collection.back();
		_collection.pop_back();
		return true;
	}

private:
	std::vector<T> _collection;
	std::atomic_flag _flag;
};

template<typename T>
class AtomicQueue
{
#define GET_LOCK \
    while (_flag.test_and_set()); \
    std::shared_ptr<int> guard(nullptr, [&](void *){_flag.clear();});

public:
	explicit AtomicQueue()
	{
		_flag.clear();
	}

	void
	push(const T &value)
	{
		GET_LOCK
		_collection.push(value);
	}

	void
	clear(void)
	{
		GET_LOCK
		_collection.clear();
	}

	int
	size()
	{
		GET_LOCK
		return _collection.size();
	}

	bool
	empty()
	{
		GET_LOCK
		return _collection.empty();
	}
	bool
	pop()
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		_collection.pop();
		return true;
	}

	bool
	pop(T &value) noexcept
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		auto elem = _collection.front();
		_collection.pop();
		return true;
	}

private:
	std::queue<T> _collection;
	std::atomic_flag _flag;
};

template<typename T>
class AtomicDequeue
{
#define GET_LOCK \
    while (_flag.test_and_set()); \
    std::shared_ptr<int> guard(nullptr, [&](void *){_flag.clear();});

public:
	explicit AtomicDequeue()
	{
		_flag.clear();
	}

	void
	pushHead(const T &value)
	{
		GET_LOCK
		_collection.push_front(value);
	}

	void
	pushTail(const T &value)
	{
		GET_LOCK
		_collection.push_back(value);
	}

	void
	clear(void)
	{
		GET_LOCK
		_collection.clear();
	}

	int
	size()
	{
		GET_LOCK
		return _collection.size();
	}

	bool
	popHead(T &value)
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		auto elem = _collection.front();
		_collection.pop_front();
		return true;
	}

	bool
	popHead()
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		_collection.pop_front();
		return true;
	}

	bool
	popTail(T &value)
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		auto elem = _collection.back();
		_collection.pop_back();
		return true;
	}

	bool
	popTail()
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		_collection.pop_back();
		return true;
	}

private:
	std::deque<T> _collection;
	std::atomic_flag _flag;
};

template<typename T>
class AtomicList
{
#define GET_LOCK \
    while (_flag.test_and_set()); \
    std::shared_ptr<int> guard(nullptr, [&](void *){_flag.clear();});

public:
	explicit AtomicList()
	{
		_flag.clear();
	}

	void
	push(const T &value)
	{
		GET_LOCK
		_collection.push_back(value);
	}

	void
	clear(void)
	{
		GET_LOCK
		_collection.clear();
	}

	int
	size()
	{
		GET_LOCK
		return _collection.size();
	}

	bool
	empty()
	{
		GET_LOCK
		return _collection.empty();
	}

	bool
	pop()
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		_collection.pop_back();
		return true;
	}

	bool
	pop(T &value) noexcept
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		value = _collection.back();
		_collection.pop_back();
		return true;
	}

private:
	std::list<T> _collection;
	std::atomic_flag _flag;
};

#endif
