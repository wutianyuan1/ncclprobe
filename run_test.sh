export LD_PRELOAD=$LD_PRELOAD:/workspace/ncclprobe/build/libncclprobe.so
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
OMPI_ALLOW_RUN_AS_ROOT=1 mpirun -np 4 ./build/test