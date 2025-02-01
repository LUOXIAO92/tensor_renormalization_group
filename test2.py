import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
info = comm.Get_info()
node = MPI.Get_processor_name()
size = comm.Get_size()

#import numpy as cp
import cupy as cp
import opt_einsum as oe
from itertools import product
import time 

ngpus = cp.cuda.runtime.getDeviceCount()
gpuid = rank % ngpus
cp.cuda.Device(gpuid).use()

for i in range(size):
    if rank == i:
        print("node:",node,"rank:",rank,"size:",size,"info:",info, "ngpus:",ngpus)
    comm.Barrier()
comm.Barrier()



K = 8
N = int(K**3)
β = 12.0
ε = 1e-4

blockU3 = 8
blockU4 = 8

#U = None
#if rank == 0:
#    U = cp.ones(shape=(2, 2, N), dtype=complex)
#comm.barrier()

from tools.trg_tools import SU2_matrix_Gauss_Legendre_quadrature, norm_for_elements, admissibility_condition
U, w, J, I = SU2_matrix_Gauss_Legendre_quadrature(K, K, K, comm, to_cupy=True)

#for i in range(U.shape[2]):
#    if i % size == rank:
#        one = oe.contract("ij,jk->ik", U[:,:,i], cp.conj(U[:,:,i]).T)
#        trone = cp.trace(one)
#        normone = cp.linalg.norm(one)
#        print(f"rank{rank}, i={i}, tr(I)={cp.abs(trone)}, norm(I)={normone**2}")
#    comm.barrier()
#comm.barrier()

from tools.mpi_tools import contract_slice
gather_slice = contract_slice(shape=(N,N), chunk=(blockU3,blockU4), comm=comm)

sendU3U4 = None
local_U3U4_shape = None
if rank == 0:
    sendU3U4 = [[] for _ in range(size)]
    for n, slices in enumerate(gather_slice):
        sU3, sU4 = slices
        sendU3U4[n % size].append([U[:,:,sU3], U[:,:,sU4]])

    #local_U3U4_shape = []
    #for r in range(size):
    #    sendU3U4[r] = cp.asarray(sendU3U4[r])
    #    local_U3U4_shape.append(sendU3U4[r].shape)
comm.barrier()


local_U3U4_shape = comm.scatter(sendobj=local_U3U4_shape, root=0)
print(rank, local_U3U4_shape)
    
t0 = time.time() if rank == 0 else None
local_U3U4 = comm.scatter(sendobj=sendU3U4, root=0)
local_b = cp.zeros(shape=(N, N), dtype=complex)
comm.barrier()
t1 = time.time() if rank == 0 else None
if rank == 0:
    print(f"Scatter data time: {t1-t0:.2e} s")
comm.barrier()

start_time = time.time()
t0 = time.time()-start_time #if rank == 0 else None
t00 = time.time()-start_time if rank == 0 else None

for n, U3U4 in enumerate(local_U3U4):
    U3, U4 = U3U4
    tr = oe.contract("abi,bcj,dck,adl->ijkl", U, U, cp.conj(U3), cp.conj(U4))
    if ε is not None:
        norm = norm_for_elements(1 - tr)
        A = (1 - tr.real) / (1 - norm / ε)
        A = cp.exp(-β * A)
        A[norm < ε] = 0.0
    else:
        A = cp.exp(-2*β * tr)
    local_b += oe.contract("ijkl,Ijkl->iI", A, cp.conj(A))
    if len(local_b[local_b == cp.inf]) != 0:
        print(f"Overflow at {n}th iteration.")
        sys.exit(0)
    if (n > 0) and (n % 25 == 0) and (rank == 0):
        t1 = time.time()-start_time  if rank == 0 else None
        print(f"Local iters:{n}. {(t1-t0) :.2e} sec/iter.")
        t0 = time.time()-start_time  if rank == 0 else None

B = comm.reduce(sendobj=local_b, 
                op=MPI.SUM,
                root=0)
comm.barrier()
t11 = time.time()-start_time if rank == 0 else None
if rank == 0:
    print(f"Tot iteration {n+1}. Time= {t11-t00:.2e} s")

del local_b

#print(B)
#if rank == 0:
#    u, s, vh = cp.linalg.svd(B)
#    print(s)
#    for ss in s:
#        print(f"{ss:.12e}")


#from init_tensor.SU2_pure_gauge import SU2_pure_gauge
#su2gauge = SU2_pure_gauge(3, 64, K, K, K, β, ε, comm, True)
#su2gauge.plaquette_tensor(64, chunk=(N,blockU3,blockU4), legs_to_hosvd=[0])