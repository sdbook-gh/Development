#include "ShmBufferProcessQueue.h"
#include <cstring>

const int BLOCK_SIZE = 4 * 1024 * 1024;

typedef ShmBufferProcessQueue<uint8_t, BLOCK_SIZE, 10, false, true> ShmqServer;

uint8_t imgBuffer[BLOCK_SIZE];

int main() {
	ShmqServer qServer("shmq");
	if (qServer.init()) {
		while(true) {
			qServer.consume([](ShmqServer::BufferBlock *pBlk){
				memcpy(imgBuffer, pBlk->buffer, pBlk->real_size);
				pBlk->real_size = 0;
				printf("consume img\n");
			});
			printf("consume\n");
			// std::this_thread::sleep_for(std::chrono::milliseconds(66));
		}
	} else {
		printf("qServer init failed\n");
	}
	return 0;
}
