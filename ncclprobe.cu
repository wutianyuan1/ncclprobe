#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <unistd.h>


using namespace std::chrono;
static bool probeInited = false;
static void* ncclLibHandle = nullptr;
static auto start_time = system_clock::now();


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
    return true;
}


// ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
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


// ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
//                       int peer, ncclComm_t comm, cudaStream_t stream)
// {

// }


// ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
//                       int peer, ncclComm_t comm, cudaStream_t stream)
// {

// }


ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream)
{
    if (!probeInited) {
        if (!initProbe()) exit(1);
    }
    auto call_time = (double)(duration_cast<milliseconds>(system_clock::now() - start_time).count()); 
    std::cout << "PID=" << getpid() << ": allreduce " << count << "bytes, at time: " << call_time << std::endl;
    using func_t = typeof(ncclAllReduce);
    auto real_func = reinterpret_cast<func_t*>(dlsym(ncclLibHandle, "ncclAllReduce"));
    return (*real_func)(sendbuff, recvbuff, count, datatype, op, comm, stream);
}
