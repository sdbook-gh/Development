#include "shmallocator.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace shmallocator {

void *shmptr{nullptr};
size_t shmsize{0};

static int32_t ptr2offset(void *ptr, void *shm_ptr) {
  if (ptr == nullptr) {
    return -1;
  }
  return (uint8_t *)ptr - (uint8_t *)shm_ptr;
}

static void *offset2ptr(int32_t offset, void *shm_ptr) {
  if (offset == -1) {
    return nullptr;
  }
  return (uint8_t *)shm_ptr + offset;
}

static void initialize_header(Header *h, size_t size, int id, bool is_first) {
  // Sanity check
  if (h == nullptr) {
    return;
  }

  h->version = SUPPORT_VERSION;
  h->prev = -1;
  h->next = -1;
  h->size = size;
  h->refcount = 0;
  h->id = id;
  h->is_free = true;
  h->bitseq = BITSEQ;

  if (is_first) {
    h->has_mutex = true;
    pthread_mutexattr_init(&(h->attr));
    pthread_mutexattr_settype(&(h->attr), PTHREAD_MUTEX_NORMAL);
    pthread_mutexattr_setpshared(&(h->attr), PTHREAD_PROCESS_SHARED);
    pthread_mutexattr_setrobust(&(h->attr), PTHREAD_MUTEX_ROBUST);
    pthread_mutex_init(&(h->mutex), &(h->attr));
  } else {
    h->has_mutex = false;
  }
}

static void destroy_header(Header *h, void *shm_ptr) {
  // Sanity check
  if (h == nullptr) {
    return;
  }

  // Adjust previous and next accordingly
  if (h->prev != -1) {
    ((Header *)offset2ptr(h->prev, shm_ptr))->next = h->next;
    // printf("prev next is %p\n", ((Header *)offset2ptr(h->prev, shm_ptr)));
  }
  if (h->next != -1) {
    ((Header *)offset2ptr(h->next, shm_ptr))->prev = h->prev;
  }

  h->version = 0;
  h->prev = -1;
  h->next = -1;
  h->size = 0;
  h->refcount = 0;
  h->id = 0;
  h->is_free = false;
  h->bitseq = 0;
}

bool initshm(const std::string &shm_file_path, size_t shm_base_address, uint32_t shm_max_size) {
  static bool res = [&] {
    int fd{0};
    bool first_create{false};
    if ((fd = open(shm_file_path.c_str(), O_RDWR)) < 0) {
      if (errno == ENOENT) {
        if ((fd = open(shm_file_path.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR)) < 0) {
          std::string str = strerror(errno);
          printf("create shmem file %s error: %s", shm_file_path.c_str(), str.c_str());
          return false;
        } else {
          first_create = true;
        }
      } else {
        std::string str = strerror(errno);
        printf("open shmem file %s error: %s", shm_file_path.c_str(), str.c_str());
        return false;
      }
    }
    if (first_create) {
      struct stat st;
      if (fstat(fd, &st) < 0) {
        std::string str = strerror(errno);
        printf("fstat shmem file %s error: %s", shm_file_path.c_str(), str.c_str());
        close(fd);
        return false;
      }
      if ((uint32_t)st.st_size < shm_max_size) {
        if (ftruncate(fd, shm_max_size) == -1) {
          std::string str = strerror(errno);
          printf("truncate shmem file %s error: %s", shm_file_path.c_str(), str.c_str());
          close(fd);
          return false;
        }
      }
    }
    void *shmem_ptr =
      mmap((void *)shm_base_address, shm_max_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
    if (shmem_ptr == MAP_FAILED) {
      std::string str = strerror(errno);
      printf("mmap shmem file %s error: %s", shm_file_path.c_str(), str.c_str());
      close(fd);
      return false;
    }
    shmptr = shmem_ptr;
    shmsize = shm_max_size;
    Header *pheader = (Header *)shmptr;
    if (first_create) {
      memset(shmptr, 0, sizeof(Header));
      int id{-1};
      shmalloc(1, &id, __FILE__, __LINE__);
      printf("shmalloc first id: %d\n", id);
    } else if (pheader->version != SUPPORT_VERSION) {
      memset(shmptr, 0, sizeof(Header));
      int id{-1};
      shmalloc(1, &id, __FILE__, __LINE__);
      printf("shmalloc reset first id: %d\n", id);
    }
    return true;
  }();
  return res;
}

void *shmalloc(uint32_t size, int *id, const char *filename, int linenumber) {
  Header *first{nullptr}, *curr{nullptr}, *best_fit{nullptr};
  uint32_t free_size{0}, best_block_size{0};

  // Verify pointers
  if (shmptr == nullptr) {
    fprintf(stderr, "%s, line %d: Shared memory is uninitialized.\n", filename, linenumber);
    return nullptr;
  }
  if (size == 0 && id != nullptr && *id == -1) {
    fprintf(stderr, "%s, line %d: Cannot allocate zero sized shared memory.\n", filename, linenumber);
    return nullptr;
  } else if (size % 2 != 0) {
    ++size;
  }
  printf("shmalloc size %u\n", size);
  if (shmsize < size + sizeof(Header)) {
    fprintf(stderr, "%s, line %d: Too big shared memory size %u to allocate.\n", filename, linenumber, size);
    return nullptr;
  }

  // Find the first header
  first = curr = (Header *)shmptr;
  best_fit = nullptr;

  // First time calling shmalloc
  if (first->bitseq != BITSEQ) {
    initialize_header(first, size, 1, 1);
    if (id != nullptr) {
      *id = 1;
    }
    first->is_free = false;
    first->refcount++;

    // Create the next header if we have enough room
    if ((free_size = (size_t)((uint8_t *)(2 * sizeof(Header)) + size)) < shmsize) {
      curr = (Header *)((uint8_t *)shmptr + sizeof(Header) + size);
      initialize_header(curr, shmsize - free_size, -1, 0);
      first->next = ptr2offset(curr, shmptr);
      curr->prev = ptr2offset(first, shmptr);
    }

    return (first + 1);
  }
  // Lock shared memory
  int ret = pthread_mutex_lock(&(first->mutex));
  if (ret == EOWNERDEAD) {
    ret = pthread_mutex_consistent(&(first->mutex));
    ret = pthread_mutex_unlock(&(first->mutex));
    ret = pthread_mutex_lock(&(first->mutex));
  }
  if (ret != 0) {
    fprintf(stderr, "%s, line %d: Cannot lock mutex in shared memory.\n", filename, linenumber);
  }
  // Loop through all headers to see if id already exists
  // Also record best spot to put this new item if it does not exist
  curr = (Header *)offset2ptr(curr->next, shmptr);
  while (curr != nullptr) {
    if (id != nullptr && *id != -1 && curr->id == (uint32_t)*id && !curr->is_free) {
      // Already have item with this id
      curr->refcount++;
      size = curr->size;
      // Can unlock mutex and return here
      pthread_mutex_unlock(&(first->mutex));
      return (curr + 1);
    }
    // Get size of this block
    if (size != 0 && (curr->size < best_block_size || best_block_size == 0) && curr->size >= size && curr->is_free) {
      best_block_size = curr->size;
      best_fit = curr;
    }
    curr = (Header *)offset2ptr(curr->next, shmptr);
  }
  // Did not find existing entry
  if (best_fit == nullptr) {
    if (size != 0) {
      // Did not find a viable chunk, failure
      fprintf(stderr,
              "%s, line %d: Shared memory ran out of available space"
              " to satisfy the request.\n",
              filename,
              linenumber);
    }
    pthread_mutex_unlock(&(first->mutex));
    return nullptr;
  }
  // Found a viable chunk - use it
  free_size = best_fit->size; // Total size of chunk before next header
  best_fit->size = size;
  best_fit->refcount = 1;
  best_fit->id = ++((Header *)shmptr)->id;
  if (id != nullptr) {
    *id = best_fit->id;
  }
  best_fit->is_free = false;
  // Check if there is enough room to make another header
  if ((free_size - best_fit->size) > sizeof(Header)) {
    curr = (Header *)((uint8_t *)best_fit + best_fit->size + sizeof(Header));
    initialize_header(curr, (size_t)(free_size - best_fit->size - sizeof(Header)), -1, 0);
    // Adjust pointers
    curr->prev = ptr2offset(best_fit, shmptr);
    curr->next = best_fit->next;
    if (best_fit->next != -1) {
      ((Header *)offset2ptr(best_fit->next, shmptr))->prev = ptr2offset(curr, shmptr);
    }
    best_fit->next = ptr2offset(curr, shmptr);
  } else {
    best_fit->size = free_size;
  }
  pthread_mutex_unlock(&(first->mutex));
  return (best_fit + 1);
}

void *shmget(int id, const char *filename, int linenumber) {
  return shmalloc(0, &id, filename, linenumber);
}

void shmfree(void *ptr, const char *filename, int linenumber) {
  Header *h, *first;
  if (ptr == nullptr) {
    // Like free(3), shmfree() of a nullptr pointer has no effect
    fprintf(stderr, "%s, line %d: free() on a nullptr pointer does nothing.\n", filename, linenumber);
    return;
  }

  // Get the associated header
  h = ((Header *)ptr) - 1;

  // More verification checks
  if (h->bitseq != BITSEQ) {
    fprintf(stderr,
            "%s, line %d: Attempted to free a pointer not allocated"
            " by shmalloc() or corruption of internal memory has "
            "occurred. Check your memory accesses.\n",
            filename,
            linenumber);
    return;
  }
  if (h->is_free) {
    fprintf(stderr,
            "%s, line %d: Attempt to shmfree() a pointer that has "
            "already been freed.\n",
            filename,
            linenumber);
    return;
  }

  // LOCK EVERYTHING
  first = (Header *)shmptr;
  int ret = pthread_mutex_lock(&(first->mutex));
  if (ret == EOWNERDEAD) {
    ret = pthread_mutex_consistent(&(first->mutex));
    ret = pthread_mutex_unlock(&(first->mutex));
    ret = pthread_mutex_lock(&(first->mutex));
  }
  if (ret != 0) {
    fprintf(stderr, "%s, line %d: Cannot lock mutex in shared memory.\n", filename, linenumber);
  }

  // If we are the last reference
  if (--(h->refcount) <= 0) {
    printf("shmfree size %u\n", h->size);
    // Adjust our size
    if (h->next != -1) {
      h->size = (uint8_t *)offset2ptr(h->next, shmptr) - (uint8_t *)h - sizeof(Header);
    } else {
      h->size = (uint8_t *)shmsize - (uint8_t *)h - sizeof(Header);
    }

    /*Check if we can delete our next to free up space*/
    if (h->next != -1 && ((Header *)offset2ptr(h->next, shmptr))->is_free) {
      h->size += (size_t)(((Header *)offset2ptr(h->next, shmptr))->size + sizeof(Header));
      destroy_header((Header *)offset2ptr(h->next, shmptr), shmptr);
    }

    // Don't delete the first entry
    if (h != first) {
      if (h->prev != -1 && ((Header *)offset2ptr(h->prev, shmptr))->is_free) {
        ((Header *)offset2ptr(h->prev, shmptr))->size += (size_t)(h->size + sizeof(Header));
        destroy_header(h, shmptr);
        h = nullptr;
      }
    }

    // Need to set h to freed
    if (h != nullptr || h == first) {
      h->is_free = true;
    }
  }
  pthread_mutex_unlock(&(first->mutex));
}

int shmgetmaxid() {
  Header *header = (Header *)shmptr;
  if (header == nullptr) {
    return -1;
  } else if (header->bitseq != BITSEQ) {
    return -1;
  }
  int ret = pthread_mutex_lock(&(header->mutex));
  if (ret == EOWNERDEAD) {
    ret = pthread_mutex_consistent(&(header->mutex));
    ret = pthread_mutex_unlock(&(header->mutex));
    ret = pthread_mutex_lock(&(header->mutex));
  }
  if (ret != 0) {
    return -1;
  }
  int max_id = header->id;
  pthread_mutex_unlock(&(header->mutex));
  return max_id;
}

bool uninitshm() {
  munmap(shmptr, shmsize);
  return true;
}

} // namespace shmallocator
