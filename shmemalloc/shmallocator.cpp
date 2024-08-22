#include "shmallocator.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace shmallocator {

void *shmptr{nullptr};
uint32_t shmsize{0};

static int32_t ptr2offset(void *ptr, void *shm_ptr) {
  if (ptr == nullptr || shm_ptr == nullptr) {
    return -1;
  }
  return (uint8_t *)ptr - (uint8_t *)shm_ptr;
}

static void *offset2ptr(int32_t offset, void *shm_ptr) {
  if (offset == -1 || shm_ptr == nullptr) {
    return nullptr;
  }
  return (uint8_t *)shm_ptr + offset;
}

void print_shmallocator() {
  if (shmptr == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return;
  }
  Header *curr = (Header *)shmptr;
  while (curr != nullptr) {
    printf("bitseq %u id %u prev %d next %d size %d free %d\n",
           curr->bitseq,
           curr->id,
           curr->prev,
           curr->next,
           curr->size,
           curr->is_free);
    curr = curr->next != -1 ? (Header *)offset2ptr(curr->next, shmptr) : nullptr;
  }
}

static void initialize_header(Header *h, size_t size, int id, bool is_first) {
  if (shmptr == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return;
  }
  // Sanity check
  if (h == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return;
  }
  // ensure hreader is not valid
  h->bitseq = 0;
  h->version = SUPPORT_VERSION;
  h->prev = -1;
  h->next = -1;
  h->size = size;
  h->id = id;
  h->is_free = true;

  if (is_first) {
    h->has_mutex = true;
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
    pthread_mutex_init(&(h->mutex), &attr);
    pthread_mutexattr_destroy(&attr);
  } else {
    h->has_mutex = false;
  }
}

static void destroy_header(Header *h, void *shm_ptr) {
  if (shmptr == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return;
  }
  // Sanity check
  if (h == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return;
  }
  if (h->bitseq != BITSEQ) {
    fprintf(stderr, "%s %d bad header bitseq\n", __FILE__, __LINE__);
    return;
  }
  Header *prev = h->prev != -1 ? (Header *)offset2ptr(h->prev, shm_ptr) : nullptr;
  if (prev != nullptr && prev->bitseq != BITSEQ) {
    fprintf(stderr, "%s %d allocation information is corrupted\n", __FILE__, __LINE__);
    print_shmallocator();
    return;
  }
  Header *next = h->next != -1 ? (Header *)offset2ptr(h->next, shm_ptr) : nullptr;
  if (next != nullptr && next->bitseq != BITSEQ) {
    fprintf(stderr, "%s %d allocation information is corrupted\n", __FILE__, __LINE__);
    print_shmallocator();
    return;
  }
  // Adjust previous and next accordingly
  // ensure hreader is not valid
  h->bitseq = 0;
  if (prev != nullptr) {
    // ensure hreader is not valid
    prev->bitseq = 0;
    prev->next = h->next;
  }
  if (next != nullptr) {
    // ensure hreader is not valid
    next->bitseq = 0;
    next->prev = h->prev;
  }
  // ensure hreader is valid
  prev->bitseq = next->bitseq = BITSEQ;
  h->version = 0;
  h->prev = -1;
  h->next = -1;
  h->size = 0;
  h->id = 0;
  h->is_free = false;
}

bool initshm(const std::string &shm_file_path, size_t shm_base_address, uint32_t shm_max_size) {
  static bool res = [&] {
    int fd{0};
    bool first_create{false};
    if ((fd = open(shm_file_path.c_str(), O_RDWR)) < 0) {
      if (errno == ENOENT) {
        if ((fd = open(shm_file_path.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR)) < 0) {
          std::string str = strerror(errno);
          fprintf(
            stderr, "%s %d create shmem file %s error: %s\n", __FILE__, __LINE__, shm_file_path.c_str(), str.c_str());
          return false;
        } else {
          first_create = true;
        }
      } else {
        std::string str = strerror(errno);
        fprintf(stderr, "%s %d open shmem file %s error: %s\n", __FILE__, __LINE__, shm_file_path.c_str(), str.c_str());
        return false;
      }
    }
    if (first_create) {
      struct stat st;
      if (fstat(fd, &st) < 0) {
        std::string str = strerror(errno);
        fprintf(
          stderr, "%s %d fstat shmem file %s error: %s\n", __FILE__, __LINE__, shm_file_path.c_str(), str.c_str());
        close(fd);
        return false;
      }
      if ((uint32_t)st.st_size < shm_max_size) {
        if (ftruncate(fd, shm_max_size) == -1) {
          std::string str = strerror(errno);
          fprintf(
            stderr, "%s %d truncate shmem file %s error: %s\n", __FILE__, __LINE__, shm_file_path.c_str(), str.c_str());
          close(fd);
          return false;
        }
      }
    }
    void *shmem_ptr =
      mmap((void *)shm_base_address, shm_max_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
    if (shmem_ptr == MAP_FAILED) {
      std::string str = strerror(errno);
      fprintf(stderr, "%s %d mmap shmem file %s error: %s\n", __FILE__, __LINE__, shm_file_path.c_str(), str.c_str());
      close(fd);
      return false;
    }
    shmptr = shmem_ptr;
    shmsize = shm_max_size;
    Header *pheader = (Header *)shmptr;
    if (first_create) {
      memset(shmptr, 0, sizeof(Header));
      int id;
      shmalloc(1, &id);
      printf("shmalloc create first id: %d\n", id);
    } else if (pheader->version != SUPPORT_VERSION) {
      memset(shmptr, 0, sizeof(Header));
      int id;
      shmalloc(1, &id);
      printf("shmalloc reset first id: %d\n", id);
    }
    // print_shmallocator();
    return true;
  }();
  return res;
}

void *shmalloc(uint32_t size, int *id) {
  if (shmptr == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return nullptr;
  }
  Header *first{nullptr}, *curr{nullptr}, *best_fit{nullptr};
  uint32_t free_size{0}, best_block_size{0};
  if (size == 0 && id != nullptr && *id == -1) {
    fprintf(stderr, "%s %d bad parameter\n", __FILE__, __LINE__);
    return nullptr;
  }
  // printf("shmalloc size %u\n", size);
  if (shmsize < size + sizeof(Header)) {
    fprintf(stderr, "%s %d size %u too big\n", __FILE__, __LINE__, size);
    return nullptr;
  }

  // Find the first header
  first = curr = (Header *)shmptr;
  best_fit = nullptr;

  // First time calling shmalloc
  if (first->bitseq != BITSEQ) {
    // no lock, since mmap file is created exclusively
    initialize_header(first, size, 1, 1);
    first->is_free = false;
    // Create the next header if we have enough room
    if ((free_size = (size_t)((uint8_t *)(2 * sizeof(Header)) + size)) < shmsize) {
      curr = (Header *)((uint8_t *)shmptr + sizeof(Header) + size);
      initialize_header(curr, shmsize - free_size, -1, 0);
      first->next = ptr2offset(curr, shmptr);
      curr->prev = ptr2offset(first, shmptr);
      // ensure hreader is valid
      curr->bitseq = BITSEQ;
    }
    // ensure hreader is valid
    first->bitseq = BITSEQ;
    if (id != nullptr) {
      *id = 1;
    }
    // print_shmallocator();
    return (first + 1);
  }
  // Lock shared memory
  int ret[3]{0};
  ret[0] = pthread_mutex_lock(&(first->mutex));
  if (ret[0] == EOWNERDEAD) {
    ret[1] = pthread_mutex_consistent(&(first->mutex));
    ret[2] = pthread_mutex_unlock(&(first->mutex));
    ret[0] = pthread_mutex_lock(&(first->mutex));
  }
  if (ret[0] != 0 || ret[1] != 0 || ret[2] != 0) {
    fprintf(stderr, "%s %d lock mutex in shared memory error\n", __FILE__, __LINE__);
    pthread_mutex_unlock(&(first->mutex));
    return nullptr;
  }
  // Loop through all headers to see if id already exists
  // Also record best spot to put this new item if it does not exist
  curr = (Header *)offset2ptr(curr->next, shmptr);
  while (curr != nullptr) {
    if (curr->bitseq != BITSEQ) {
      fprintf(stderr, "%s %d allocation information is corrupted\n", __FILE__, __LINE__);
      print_shmallocator();
      pthread_mutex_unlock(&(first->mutex));
      return nullptr;
    }
    if (id != nullptr && *id != -1 && curr->id == (uint32_t)*id && !curr->is_free) {
      // Already have item with this id
      pthread_mutex_unlock(&(first->mutex));
      // print_shmallocator();
      return (curr + 1);
    }
    // Get size of this block
    if (size != 0 && (curr->size < best_block_size || best_block_size == 0) && curr->size >= size && curr->is_free) {
      best_block_size = curr->size;
      best_fit = curr;
    }
    curr = curr->next != -1 ? (Header *)offset2ptr(curr->next, shmptr) : nullptr;
  }
  // Did not find existing entry
  if (best_fit == nullptr) {
    if (size != 0) {
      // Did not find a viable chunk, failure
      fprintf(stderr, "%s %d no enough free space to allocate\n", __FILE__, __LINE__);
    }
    pthread_mutex_unlock(&(first->mutex));
    return nullptr;
  }
  // Found a viable chunk - use it
  // ensure hreader is not valid
  best_fit->bitseq = 0;
  free_size = best_fit->size; // Total size of chunk before next header
  best_fit->size = size;
  best_fit->id = ++((Header *)shmptr)->id;
  best_fit->is_free = false;
  Header *next = best_fit->next != -1 ? (Header *)offset2ptr(best_fit->next, shmptr) : nullptr;
  if (next != nullptr && next->bitseq != BITSEQ) {
    fprintf(stderr, "%s %d allocation information is corrupted\n", __FILE__, __LINE__);
    print_shmallocator();
    pthread_mutex_unlock(&(first->mutex));
    return nullptr;
  }
  // Check if there is enough room to make another header
  if ((free_size - best_fit->size) > sizeof(Header)) {
    curr = (Header *)((uint8_t *)best_fit + best_fit->size + sizeof(Header));
    initialize_header(curr, (size_t)(free_size - best_fit->size - sizeof(Header)), -1, 0);
    // Adjust pointers
    curr->prev = ptr2offset(best_fit, shmptr);
    curr->next = best_fit->next;
    // ensure hreader is valid
    curr->bitseq = BITSEQ;
    if (next != nullptr) {
      // ensure hreader is not valid
      next->bitseq = 0;
      next->prev = ptr2offset(curr, shmptr);
      // ensure hreader is valid
      next->bitseq = BITSEQ;
    }
    best_fit->next = ptr2offset(curr, shmptr);
  } else {
    best_fit->size = free_size;
    best_fit->next = -1;
  }
  // ensure hreader is valid
  best_fit->bitseq = BITSEQ;
  if (id != nullptr) {
    *id = best_fit->id;
  }
  pthread_mutex_unlock(&(first->mutex));
  // print_shmallocator();
  return (best_fit + 1);
}

void *shmget(int id) {
  return shmalloc(0, &id);
}

void shmfree(void *ptr) {
  return;
  if (shmptr == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return;
  }
  Header *h{nullptr};
  Header *first{nullptr};
  if (ptr == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return;
  }
  h = ((Header *)ptr) - 1;
  if (h->bitseq != BITSEQ) {
    fprintf(stderr, "%s %d allocation information is corrupted\n", __FILE__, __LINE__);
    print_shmallocator();
    return;
  }
  Header *prev = h->prev != -1 ? (Header *)offset2ptr(h->prev, shmptr) : nullptr;
  if (prev != nullptr && prev->bitseq != BITSEQ) {
    fprintf(stderr, "%s %d allocation information is corrupted\n", __FILE__, __LINE__);
    print_shmallocator();
    return;
  }
  Header *next = h->next != -1 ? (Header *)offset2ptr(h->next, shmptr) : nullptr;
  if (next != nullptr && next->bitseq != BITSEQ) {
    fprintf(stderr, "%s %d allocation information is corrupted\n", __FILE__, __LINE__);
    print_shmallocator();
    return;
  }
  if (h->is_free) {
    fprintf(stderr, "%s %d double free error\n", __FILE__, __LINE__);
    return;
  }

  first = (Header *)shmptr;
  int ret[3]{0};
  ret[0] = pthread_mutex_lock(&(first->mutex));
  if (ret[0] == EOWNERDEAD) {
    ret[1] = pthread_mutex_consistent(&(first->mutex));
    ret[2] = pthread_mutex_unlock(&(first->mutex));
    ret[0] = pthread_mutex_lock(&(first->mutex));
  }
  if (ret[0] != 0 || ret[1] != 0 || ret[2] != 0) {
    fprintf(stderr, "%s %d lock mutex in shared memory error\n", __FILE__, __LINE__);
    pthread_mutex_unlock(&(first->mutex));
    return;
  }
  // printf("shmfree size %u\n", h->size);
  // ensure hreader is not valid
  h->bitseq = 0;
  // Adjust our size
  if (next != nullptr) {
    h->size = (uint8_t *)next - (uint8_t *)h - sizeof(Header);
  } else {
    h->size = (uint8_t *)shmptr + shmsize - (uint8_t *)h - sizeof(Header);
  }
  /*Check if we can delete our next to free up space*/
  if (next != nullptr && next->is_free) {
    h->size += next->size + sizeof(Header);
    destroy_header(next, shmptr);
    h->bitseq = BITSEQ;
  } else if (prev != nullptr && prev->is_free) {
    // ensure hreader is not valid
    prev->bitseq = 0;
    prev->size += h->size + sizeof(Header);
    destroy_header(h, shmptr);
    // ensure hreader is valid
    prev->bitseq = BITSEQ;
  } else {
    h->bitseq = BITSEQ;
  }
  pthread_mutex_unlock(&(first->mutex));
  print_shmallocator();
}

int shmgetmaxid() {
  if (shmptr == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return -1;
  }
  Header *first = (Header *)shmptr;
  if (first->bitseq != BITSEQ) {
    fprintf(stderr, "%s %d allocation information is corrupted\n", __FILE__, __LINE__);
    print_shmallocator();
    return -1;
  }
  int ret[3]{0};
  ret[0] = pthread_mutex_lock(&(first->mutex));
  if (ret[0] == EOWNERDEAD) {
    ret[1] = pthread_mutex_consistent(&(first->mutex));
    ret[2] = pthread_mutex_unlock(&(first->mutex));
    ret[0] = pthread_mutex_lock(&(first->mutex));
  }
  if (ret[0] != 0 || ret[1] != 0 || ret[2] != 0) {
    fprintf(stderr, "%s %d lock mutex in shared memory error\n", __FILE__, __LINE__);
    pthread_mutex_unlock(&(first->mutex));
    return -1;
  }
  int max_id = first->id;
  pthread_mutex_unlock(&(first->mutex));
  return max_id;
}

bool uninitshm() {
  if (shmptr == nullptr) {
    fprintf(stderr, "%s %d nullptr\n", __FILE__, __LINE__);
    return false;
  }
  Header *first = (Header *)shmptr;
  if (first->bitseq != BITSEQ) {
    fprintf(stderr, "%s %d allocation information is corrupted\n", __FILE__, __LINE__);
    print_shmallocator();
    return false;
  }
  int ret[3]{0};
  ret[0] = pthread_mutex_lock(&(first->mutex));
  if (ret[0] == EOWNERDEAD) {
    ret[1] = pthread_mutex_consistent(&(first->mutex));
    ret[2] = pthread_mutex_unlock(&(first->mutex));
    ret[0] = pthread_mutex_lock(&(first->mutex));
  }
  if (ret[0] != 0 || ret[1] != 0 || ret[2] != 0) {
    fprintf(stderr, "%s %d lock mutex in shared memory error\n", __FILE__, __LINE__);
    pthread_mutex_unlock(&(first->mutex));
    return false;
  }
  // munmap(shmptr, shmsize);
  shmptr = nullptr;
  pthread_mutex_unlock(&(first->mutex));
  return true;
}

} // namespace shmallocator
