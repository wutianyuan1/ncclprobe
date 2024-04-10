#pragma once
#include <boost/preprocessor.hpp>


#define X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE(r, data, elem)    \
    case elem : return BOOST_PP_STRINGIZE(elem);

#define DEFINE_ENUM_WITH_STRING_CONVERSIONS(name, enumerators)                \
    enum name {                                                               \
        BOOST_PP_SEQ_ENUM(enumerators)                                        \
    };                                                                        \
                                                                              \
    inline const char* ToString(name v)                                       \
    {                                                                         \
        switch (v)                                                            \
        {                                                                     \
            BOOST_PP_SEQ_FOR_EACH(                                            \
                X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE,          \
                name,                                                         \
                enumerators                                                   \
            )                                                                 \
            default: return "[Unknown " BOOST_PP_STRINGIZE(name) "]";         \
        }                                                                     \
    }

DEFINE_ENUM_WITH_STRING_CONVERSIONS(
    NcclNumber, 
    (SEND)
    (RECV)
    (BCAST)
    (BROADCAST)
    (ALL_GATHER)
    (REDUCE_SCATTER)
    (ALL_REDUCE)
)

// max length of PCIe string
#define PCI_STR_LEN 64
// max #GPUs
#define MAX_DEVS 4
// 7 metadata fields: numfields, maxrecords, numrecords, head, tail, magic, last_event
#define METADATA_FIELDS  7
// buffer magic number
#define BUFFER_MAGIC 0xdeadbeef
// max capacity of the buffer
#define BUFFER_SIZE 20000
