/* Generated by the protocol buffer compiler.  DO NOT EDIT! */
/* Generated from: amessage.proto */

#ifndef PROTOBUF_C_amessage_2eproto__INCLUDED
#define PROTOBUF_C_amessage_2eproto__INCLUDED

#include <protobuf-c/protobuf-c.h>

PROTOBUF_C__BEGIN_DECLS

#if PROTOBUF_C_VERSION_NUMBER < 1000000
# error This file was generated by a newer version of protoc-c which is incompatible with your libprotobuf-c headers. Please update your headers.
#elif 1005000 < PROTOBUF_C_MIN_COMPILER_VERSION
# error This file was generated by an older version of protoc-c which is incompatible with your libprotobuf-c headers. Please regenerate this file with a newer version of protoc-c.
#endif


typedef struct Test__Protobuf__C__Submessage1 Test__Protobuf__C__Submessage1;
typedef struct Test__Protobuf__C__Submessage2 Test__Protobuf__C__Submessage2;
typedef struct Test__Protobuf__C__AMessage Test__Protobuf__C__AMessage;


/* --- enums --- */


/* --- messages --- */

struct  Test__Protobuf__C__Submessage1
{
  ProtobufCMessage base;
  int32_t value;
};
#define TEST__PROTOBUF__C__SUBMESSAGE1__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&test__protobuf__c__submessage1__descriptor) \
    , 0 }


struct  Test__Protobuf__C__Submessage2
{
  ProtobufCMessage base;
  int32_t value;
};
#define TEST__PROTOBUF__C__SUBMESSAGE2__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&test__protobuf__c__submessage2__descriptor) \
    , 0 }


struct  Test__Protobuf__C__AMessage
{
  ProtobufCMessage base;
  int32_t a;
  protobuf_c_boolean has_b;
  int32_t b;
  size_t n_c;
  int32_t *c;
  size_t n_d;
  char **d;
  ProtobufCBinaryData e;
  Test__Protobuf__C__Submessage1 *f;
  size_t n_g;
  Test__Protobuf__C__Submessage2 **g;
};
#define TEST__PROTOBUF__C__AMESSAGE__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&test__protobuf__c__amessage__descriptor) \
    , 0, 0, 0, 0,NULL, 0,NULL, {0,NULL}, NULL, 0,NULL }


/* Test__Protobuf__C__Submessage1 methods */
void   test__protobuf__c__submessage1__init
                     (Test__Protobuf__C__Submessage1         *message);
size_t test__protobuf__c__submessage1__get_packed_size
                     (const Test__Protobuf__C__Submessage1   *message);
size_t test__protobuf__c__submessage1__pack
                     (const Test__Protobuf__C__Submessage1   *message,
                      uint8_t             *out);
size_t test__protobuf__c__submessage1__pack_to_buffer
                     (const Test__Protobuf__C__Submessage1   *message,
                      ProtobufCBuffer     *buffer);
Test__Protobuf__C__Submessage1 *
       test__protobuf__c__submessage1__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   test__protobuf__c__submessage1__free_unpacked
                     (Test__Protobuf__C__Submessage1 *message,
                      ProtobufCAllocator *allocator);
/* Test__Protobuf__C__Submessage2 methods */
void   test__protobuf__c__submessage2__init
                     (Test__Protobuf__C__Submessage2         *message);
size_t test__protobuf__c__submessage2__get_packed_size
                     (const Test__Protobuf__C__Submessage2   *message);
size_t test__protobuf__c__submessage2__pack
                     (const Test__Protobuf__C__Submessage2   *message,
                      uint8_t             *out);
size_t test__protobuf__c__submessage2__pack_to_buffer
                     (const Test__Protobuf__C__Submessage2   *message,
                      ProtobufCBuffer     *buffer);
Test__Protobuf__C__Submessage2 *
       test__protobuf__c__submessage2__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   test__protobuf__c__submessage2__free_unpacked
                     (Test__Protobuf__C__Submessage2 *message,
                      ProtobufCAllocator *allocator);
/* Test__Protobuf__C__AMessage methods */
void   test__protobuf__c__amessage__init
                     (Test__Protobuf__C__AMessage         *message);
size_t test__protobuf__c__amessage__get_packed_size
                     (const Test__Protobuf__C__AMessage   *message);
size_t test__protobuf__c__amessage__pack
                     (const Test__Protobuf__C__AMessage   *message,
                      uint8_t             *out);
size_t test__protobuf__c__amessage__pack_to_buffer
                     (const Test__Protobuf__C__AMessage   *message,
                      ProtobufCBuffer     *buffer);
Test__Protobuf__C__AMessage *
       test__protobuf__c__amessage__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   test__protobuf__c__amessage__free_unpacked
                     (Test__Protobuf__C__AMessage *message,
                      ProtobufCAllocator *allocator);
/* --- per-message closures --- */

typedef void (*Test__Protobuf__C__Submessage1_Closure)
                 (const Test__Protobuf__C__Submessage1 *message,
                  void *closure_data);
typedef void (*Test__Protobuf__C__Submessage2_Closure)
                 (const Test__Protobuf__C__Submessage2 *message,
                  void *closure_data);
typedef void (*Test__Protobuf__C__AMessage_Closure)
                 (const Test__Protobuf__C__AMessage *message,
                  void *closure_data);

/* --- services --- */


/* --- descriptors --- */

extern const ProtobufCMessageDescriptor test__protobuf__c__submessage1__descriptor;
extern const ProtobufCMessageDescriptor test__protobuf__c__submessage2__descriptor;
extern const ProtobufCMessageDescriptor test__protobuf__c__amessage__descriptor;

PROTOBUF_C__END_DECLS


#endif  /* PROTOBUF_C_amessage_2eproto__INCLUDED */