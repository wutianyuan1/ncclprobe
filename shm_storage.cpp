#include "shm_storage.hpp"
#include <iostream>
#include <cassert>
using namespace boost::interprocess;

// numfields, maxrecords, numrecords, head, tail
#define METADATA_FIELDS  5


/* === Buffer Implementation === */
RecordBuffer::RecordBuffer(size_t numFields, size_t maxRecords, void* mem_address)
    : numFields(numFields), maxRecords(maxRecords), numRecords(0), head(0), tail(0)
{
    this->addr = reinterpret_cast<uint64_t*>(mem_address);
    this->buffer = this->addr + METADATA_FIELDS;
    updateMeta();
}

RecordBuffer::~RecordBuffer() {}

void RecordBuffer::updateMeta()
{
    addr[0] = numFields;
    addr[1] = maxRecords;
    addr[2] = numRecords;
    addr[3] = head;
    addr[4] = tail;
}

void RecordBuffer::push(std::vector<uint64_t>&& record)
{
    assert(record.size() == numFields);
    memcpy(buffer + tail * numFields, record.data(), numFields * sizeof(uint64_t));
    tail = (tail + 1) % maxRecords;
    if (numRecords == maxRecords)
        head = (head + 1) % maxRecords;
    if (numRecords < maxRecords)
        numRecords++;
    updateMeta();
}

void RecordBuffer::push(std::vector<uint64_t>& record)
{
    this->push(std::move(record));
}

inline bool RecordBuffer::full() noexcept
{
    return numRecords == maxRecords;
}

inline bool RecordBuffer::empty() noexcept
{
    return numRecords == 0;
}


/* === Storage Implementation === */
NcclRecordStorage::NcclRecordStorage(size_t numFields, size_t maxRecords)
    : numFields(numFields), maxRecords(maxRecords)
{
    shm = shared_memory_object(open_or_create, "ncclRecord", read_write);
    shm.truncate((METADATA_FIELDS + numFields * maxRecords) * sizeof(uint64_t));
    region = mapped_region(shm, read_write);
    buffer = RecordBuffer(numFields, maxRecords, region.get_address());
}

NcclRecordStorage::~NcclRecordStorage()
{
    shared_memory_object::remove("ncclRecord");
    std::cout << "Shutdown gracefully!" << std::endl;
}

void NcclRecordStorage::addRecord(std::vector<uint64_t>&& record)
{
    buffer.push(std::move(record));
}

void NcclRecordStorage::addRecord(std::vector<uint64_t>& record)
{
    buffer.push(record);
}


