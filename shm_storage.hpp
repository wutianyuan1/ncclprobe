#include <vector>
#include <atomic>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>


class RecordBuffer
{
    size_t numFields;
    size_t maxRecords;
    size_t numRecords;
    off_t head;
    off_t tail;
    uint64_t* buffer;
    uint64_t* addr;
public:
    RecordBuffer() = default;
    RecordBuffer(size_t numFields, size_t maxRecords, void* mem_address);
    ~RecordBuffer();
    void push(std::vector<uint64_t>&& record);
    void push(std::vector<uint64_t>& record);
    bool full() noexcept;
    bool empty() noexcept;
private:
    void loadMeta();
    void updateMeta();
};


struct Lock
{
    boost::interprocess::interprocess_mutex  mutex;
};


class NcclRecordStorage
{
    boost::interprocess::shared_memory_object shm;
    boost::interprocess::mapped_region region;
    boost::interprocess::shared_memory_object lock_shm;
    boost::interprocess::mapped_region lock_region;
    Lock* lock;
    size_t numFields;
    size_t maxRecords;
    RecordBuffer buffer;
public:
    NcclRecordStorage() = default;
    NcclRecordStorage(size_t numFields, size_t maxRecords);
    ~NcclRecordStorage();
    void addRecord(std::vector<uint64_t>&& record);
    void addRecord(std::vector<uint64_t>& record);
};
