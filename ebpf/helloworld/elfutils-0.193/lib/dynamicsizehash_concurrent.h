/* Copyright (C) 2000-2019 Red Hat, Inc.
   This file is part of elfutils.
   Written by Srdan Milakovic <sm108@rice.edu>, 2019.
   Derived from Ulrich Drepper <drepper@redhat.com>, 2000.

   This file is free software; you can redistribute it and/or modify
   it under the terms of either

     * the GNU Lesser General Public License as published by the Free
       Software Foundation; either version 3 of the License, or (at
       your option) any later version

   or

     * the GNU General Public License as published by the Free
       Software Foundation; either version 2 of the License, or (at
       your option) any later version

   or both in parallel, as here.

   elfutils is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received copies of the GNU General Public License and
   the GNU Lesser General Public License along with this program.  If
   not, see <http://www.gnu.org/licenses/>.  */

#include <stddef.h>
#include <pthread.h>
#include <stdatomic.h>
/* Before including this file the following macros must be defined:

   NAME      name of the hash table structure.
   TYPE      data type of the hash table entries

   The following macros if present select features:

   ITERATE   iterating over the table entries is possible
   HASHTYPE  integer type for hash values, default unsigned long int
 */



#ifndef HASHTYPE
# define HASHTYPE unsigned long int
#endif

#ifndef RESIZE_BLOCK_SIZE
# define RESIZE_BLOCK_SIZE 256
#endif

/* Defined separately.  */
extern size_t next_prime (size_t seed);


/* Table entry type.  */
#define _DYNHASHCONENTTYPE(name)       \
  typedef struct name##_ent         \
  {                                 \
    _Atomic(HASHTYPE) hashval;      \
    atomic_uintptr_t val_ptr;       \
  } name##_ent
#define DYNHASHENTTYPE(name) _DYNHASHCONENTTYPE (name)
DYNHASHENTTYPE (NAME);

/* Type of the dynamic hash table data structure.  */
#define _DYNHASHCONTYPE(name) \
typedef struct                                     \
{                                                  \
  size_t size;                                     \
  size_t old_size;                                 \
  atomic_size_t filled;                            \
  name##_ent *table;                               \
  name##_ent *old_table;                           \
  atomic_size_t resizing_state;                    \
  atomic_size_t next_init_block;                   \
  atomic_size_t num_initialized_blocks;            \
  atomic_size_t next_move_block;                   \
  atomic_size_t num_moved_blocks;                  \
  pthread_rwlock_t resize_rwl;                     \
} name
#define DYNHASHTYPE(name) _DYNHASHCONTYPE (name)
DYNHASHTYPE (NAME);



#define _FUNCTIONS(name)                                            \
/* Initialize the hash table.  */                                   \
extern int name##_init (name *htab, size_t init_size);              \
                                                                    \
/* Free resources allocated for hash table.  */                     \
extern int name##_free (name *htab);                                \
                                                                    \
/* Insert new entry.  */                                            \
extern int name##_insert (name *htab, HASHTYPE hval, TYPE data);    \
                                                                    \
/* Find entry in hash table.  */                                    \
extern TYPE name##_find (name *htab, HASHTYPE hval);
#define FUNCTIONS(name) _FUNCTIONS (name)
FUNCTIONS (NAME)


#ifndef NO_UNDEF
# undef DYNHASHENTTYPE
# undef DYNHASHTYPE
# undef FUNCTIONS
# undef _FUNCTIONS
# undef XFUNCTIONS
# undef _XFUNCTIONS
# undef NAME
# undef TYPE
# undef ITERATE
# undef COMPARE
# undef FIRST
# undef NEXT
#endif
