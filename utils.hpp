#pragma once
#include <string>
#include <cmath>

// floating point comparison
#define FLOAT_EQ(a, b) (fabs((a) - (b)) < 1e-4)

enum DistEngine{
    undefined = 0,
    mpi = 1,
    torch_run = 2,
    auto_find = 3
};


inline int get_int_value_from_env(DistEngine dist, const char* mpi_choice, const char* torchrun_choice)
{
    if (dist == DistEngine::mpi)
        return std::atoi(getenv(mpi_choice));
    else if (dist == DistEngine::torch_run)
        return std::atoi(getenv(torchrun_choice));
    else if (dist == DistEngine::auto_find)
        return getenv(mpi_choice) ?\
            std::atoi(getenv(mpi_choice)) : std::atoi(getenv(torchrun_choice));
    else  // default: "torchrun_choice" from env
        return std::atoi(getenv(torchrun_choice));
}


inline int get_rank(DistEngine dist)
{
    return get_int_value_from_env(dist, "OMPI_COMM_WORLD_RANK", "RANK");
}

inline int get_local_rank(DistEngine dist)
{
    return get_int_value_from_env(dist, "OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK");
}

inline int get_world_size(DistEngine dist)
{
    return get_int_value_from_env(dist, "OMPI_COMM_WORLD_SIZE", "WORLD_SIZE");
}

inline int get_local_world_size(DistEngine dist)
{
    return get_int_value_from_env(dist, "OMPI_COMM_WORLD_LOCAL_SIZE", "LOCAL_WORLD_SIZE");
}

inline const char* get_nccl_path(const char* nccl_path = nullptr)
{
    return getenv(nccl_path ? nccl_path : "NCCL_PATH");
}