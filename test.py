from mpi4py import MPI
from itertools import chain, product
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node = MPI.Get_processor_name()
info = comm.Get_info()


K = 8
N = int(K**3)
ε = 0.1

blockU3 = 8
blockU4 = 8
iteration = product(range(0, N, blockU3), range(0, N, blockU4))


from tools.mpi_tools import contraction_slicer, slice_info_distributer, gather_slice_info

t0 = time.time() if rank == 0 else None
slicing = contraction_slicer(shape=(N,N), chunk=(blockU3,blockU4))
comm.barrier()
t1 = time.time() if rank == 0 else None
if rank == 0:
    print(f"slicing legs finished. time= {t1-t0:.2} s")

t0 = time.time() if rank == 0 else None
slice_info = slice_info_distributer(slicing)
comm.barrier()
t1 = time.time() if rank == 0 else None
if rank == 0:
    print(f"distribute slicing info finished. time= {t1-t0:.2} s")
comm.barrier()

ones = np.ones(shape=(N,N), dtype=int)

t0 = time.time() if rank == 0 else None
gather_slice = gather_slice_info(slice_info)
comm.barrier()
t1 = time.time() if rank == 0 else None
if rank == 0:
    print(f"gather slicing info finished. time= {t1-t0:.2} s")

#for i in range(size):
#
#    t0 = time.time() if rank == 0 else None
#    if rank == i:
#        comm.send([slice_info], dest=0, tag=i)
#        print(f"send slice info from rank:{rank} to rank:0")
#    comm.barrier()
#    t1 = time.time() if rank == 0 else None
#    if rank == 0:
#        print(f"send slice info from rank:{i} to rank:0 finished. time= {t1-t0:.2} s")
#    comm.barrier()
#
#    t0 = time.time() if rank == 0 else None
#    if rank == 0:
#        slice_info_recv = comm.recv(source=i, tag=i)
#        slice_info_recv = slice_info_recv[0]
#        print(f"recv slice info from rank:{i} to rank:{rank}.")
#        summation = 0
#        for s in slice_info_recv:
#            summation += np.sum(ones[s])
#        print(summation)
#    comm.barrier()
#    t1 = time.time() if rank == 0 else None
#    if rank == 0:
#        print(f"recv slice info from rank:{i} to rank:0 finished. time= {t1-t0:.2} s")
#    comm.barrier()