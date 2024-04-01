#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <unistd.h>
#include <memory>
#include "shm_storage.hpp"


using namespace std::chrono;
static bool probeInited = false;
static void* ncclLibHandle = nullptr;
static auto start_time = system_clock::now();
std::shared_ptr<NcclRecordStorage> storage_buffer = nullptr;


bool initProbe(){
    char* ncclPath = getenv("NCCL_PATH");
    if (!ncclPath) ncclPath = (char*)"/usr/lib/x86_64-linux-gnu/libnccl.so.2";
    void* handle = dlopen(ncclPath, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error: " << dlerror() << std::endl;
        return false;
    }
    dlerror();
    ncclLibHandle = handle;
    probeInited = true;
    storage_buffer = std::shared_ptr<NcclRecordStorage>(new NcclRecordStorage(7, 1000));
    return true;
}


ncclResult_t probeBegin(const void* buff1, const void* buff2, size_t count,
                        ncclDataType_t datatype, ncclComm_t comm,
                        cudaStream_t stream, int aux)
{
    uint64_t buff1Addr = reinterpret_cast<uint64_t>(buff1),
             buff2Addr = reinterpret_cast<uint64_t>(buff2),
             dtypeID = (uint64_t)(datatype),
             pid = (uint64_t)(getpid());
    auto call_time = (double)(duration_cast<milliseconds>(system_clock::now() - start_time).count()); 
    int dev_id = -1;
    char pcistr[50];
    cudaGetDevice(&dev_id);
    cudaDeviceGetPCIBusId(pcistr, 50, dev_id);
    // std::cout << "PID=" << getpid() << ": allreduce " << count << "bytes, at time: " << call_time\
    //     << " device id: " << dev_id  << " pci: " << pcistr << std::endl;
    storage_buffer->addRecord(std::vector<uint64_t>({
        pid, (uint64_t)0, count, (uint64_t)(call_time * 1000), (uint64_t)(dev_id), buff1Addr, buff2Addr
    }));
    return ncclSuccess;
}

// ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
//                       int peer, ncclComm_t comm, cudaStream_t stream)
// {

// }


// ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
//                       int peer, ncclComm_t comm, cudaStream_t stream)
// {

// }


// ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
//                       int root, ncclComm_t comm, cudaStream_t stream)
// {

// }


// ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
//                            ncclDataType_t datatype, int root,
//                            ncclComm_t comm, cudaStream_t stream)
// {

// }


// ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
//                            ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
// {

// }


// ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
//                                ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
//                                cudaStream_t stream)
// {

// }


ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream)
{
    if (!probeInited) {
        if (!initProbe()) exit(1);
    }
    using func_t = typeof(ncclAllReduce);
    auto real_func = reinterpret_cast<func_t*>(dlsym(ncclLibHandle, "ncclAllReduce"));
    probeBegin(sendbuff, recvbuff, count, datatype, comm, stream, 0);
    auto ret = (*real_func)(sendbuff, recvbuff, count, datatype, op, comm, stream);
    return ret;
}
