#ifndef UPLOAD_VEHICLE_REPORT_INCLUDE_ATOMICDEQUEUE_H
#define UPLOAD_VEHICLE_REPORT_INCLUDE_ATOMICDEQUEUE_H

#include <atomic>
#include <memory>
#include "Utils.h"

namespace v2x
{
template<typename T>
class AtomicDequeue
{
#define GET_LOCK \
    Guard _guard([&](){while (_flag.test_and_set());}, [&](){_flag.clear();});

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
	popHead(T &value) noexcept
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		value = _collection.front();
		_collection.pop_front();
		return true;
	}

	bool
	popHead() noexcept
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		_collection.pop_front();
		return true;
	}

	bool
	popTail(T &value) noexcept
	{
		GET_LOCK
		if (_collection.empty())
			return false;
		value = _collection.back();
		_collection.pop_back();
		return true;
	}

	bool
	popTail() noexcept
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

}
#endif //UPLOAD_VEHICLE_REPORT_INCLUDE_ATOMICDEQUEUE_H
