#import cupyx.distributed as dist
#from cupyx.distributed import NCCLBackend, array
from mpi4py import MPI

world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_size = world_comm.Get_size()

info = world_comm.Get_info()
node = MPI.Get_processor_name()

#sharedcomm = world_comm.Split_type(MPI.COMM_TYPE_SHARED)
#local_rank = sharedcomm.Get_rank()
#local_size = sharedcomm.Get_size()


cart_comm = world_comm.Create_cart(dims=[2,3])
cart_coor = cart_comm.Get_coords(world_rank)
cart_rank = cart_comm.Get_cart_rank(cart_coor)



for i in range(world_size):
    if world_rank == i:
        #print("node:",node,"world_rank:",world_rank,"world_size:",world_size,"info:",info)
        #print("node:",node,
        #      "world_rank:",world_rank,"world_size:",world_size,
        #      "local_rank", local_rank,"local_size", local_size,
        #      "MPI.COMM_TYPE_SHARED", MPI.COMM_TYPE_SHARED)
        print("node:",node,
              "world_rank:",world_rank,"world_size:",world_size,
              "cart_rank", cart_rank,"cart_coor", cart_coor)

        
    world_comm.Barrier()
world_comm.Barrier()


import cupy as cp
import opt_einsum as oe

K = 16
N = int(K**3)
ε = 0.1

blockU3 = 8
blockU4 = 8

#U = cp.ones(shape=(2, 2, N), dtype=complex)

#def process_0():
#    import cupyx.distributed
#    cp.cuda.Device(0).use()
#    comm = cupyx.distributed.init_process_group(2, 0)
#    array = cp.ones(1)
#    comm.broadcast(array, 0)
#
#def process_1():
#    import cupyx.distributed
#    cp.cuda.Device(1).use()
#    comm = cupyx.distributed.init_process_group(2, 1)
#    array = cp.zeros(1)
#    comm.broadcast(array, 0)
#    cp.equal(array, cp.ones(1))
#
#process_0()
#process_1()