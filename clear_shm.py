from multiprocessing import shared_memory

shm_size = (7 * 1000 + 5) * 8
shm = shared_memory.SharedMemory("ncclRecord", create=False, size=shm_size)
shm.close()
shm.unlink()