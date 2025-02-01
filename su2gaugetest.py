import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
info = comm.Get_info()
node = MPI.Get_processor_name()
size = comm.Get_size()

for i in range(size):
    if rank == i:
        print("node:",node,"rank:",rank,"size:",size,"info:",info)
    comm.Barrier()
comm.Barrier()

import numpy as np
import cupy as cp
import opt_einsum as oe
from itertools import product
import time 

K = 6
N = int(K**3)
β = 40.96
ε = 1e-5

ε = None if ε < 1e-13 else ε

blockU3 = 6
blockU4 = 6

#ngpus = cp.cuda.runtime.getDeviceCount()
#gpuid = rank % ngpus
#cp.cuda.Device(gpuid).use()
from tools.mpi_tools import use_gpu
use_gpu(usegpu=True, comm=comm)

from SU2_pure_gauge.SU2_pure_gauge_init import SU2_pure_gauge
su2gauge = SU2_pure_gauge(3, 64, K, K, K, β, ε, comm=comm, use_gpu=True)
A = su2gauge.plaquette_tensor(64, chunk=(N,K,K), legs_to_hosvd=[0])

from tools.trg_tools import env_tensor_for_3d_SU2_pure_gauge
T = env_tensor_for_3d_SU2_pure_gauge(A, 't', comm, use_gpu=True)

if rank == 0:
    print(oe.contract("xyxy", A))