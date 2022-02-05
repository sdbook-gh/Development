#include "ShmBufferProcessQueue.h"
#include <cstring>
#include <chrono>
#include <thread>

const int BLOCK_SIZE = 4 * 1024 * 1024;

typedef ShmBufferProcessQueue<uint8_t, BLOCK_SIZE, 10, true, true> ShmqServer;

uint8_t imgBuffer[BLOCK_SIZE];

int main() {
	ShmqServer qServer("shmq");
	if (qServer.init()) {
		while(true) {
			qServer.produce([](ShmqServer::BufferBlock *pBlk){
				memcpy(pBlk->buffer, imgBuffer, sizeof(imgBuffer));
				pBlk->real_size = sizeof(imgBuffer);
				printf("produce img\n");
				std::this_thread::sleep_for(std::chrono::milliseconds(66));
			});
		}
	} else {
		printf("qServer init failed\n");
	}
	return 0;
}
