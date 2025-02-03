from mpi4py import MPI
from itertools import chain, product
from functools import reduce
import opt_einsum as oe
import cupy as cp
import math

#MPI_COMM = MPI.COMM_WORLD
#MPI_SIZE = MPI_COMM.Get_size()
#MPI_RANK = MPI_COMM.Get_rank()
#MPI_INFO = MPI_COMM.Get_info()
MPI_NODE = MPI.Get_processor_name()

def gpu_syn(gpu):
    if gpu:
        cp.cuda.get_current_stream().synchronize()

def use_gpu(usegpu:bool, comm:MPI.Intercomm):
    if usegpu:
        ngpus = cp.cuda.runtime.getDeviceCount()
        gpuid = comm.Get_rank() % ngpus
        cp.cuda.Device(gpuid).use()

def pure_gauge_slice(shape:tuple, chunk:tuple, tolist=False):
    """
    Slice the summation that used to compute the hosvd of plaquette 
    P_{U0, U1, U2, U3}

    Parameters
    ----------
    `shape` : Shape of legs of plaquette.

    `chunk` : Devide the leg (U0, U1, U2, U3) into chunk=(chunkU0, chunkU1, chunkU2, chunkU3).
    
    `tolist` : Convert the iterator to list.

    ----------

    Retruns
    -------
    An iterator of combination of slices. 

    -------

    Example
    -------
    input : `shape`=(N0, N1, N2, N3), `chunk`=(N0, N1, dU2, dU3)
    return : [ [slice(0,N0), slice(0,N1), slice(0,dU2), slice(0,dU3)], 
               [slice(0,N0), slice(0,N1), slice(0,dU2), slice(dU3,2*dU3)], ..., 
               [slice(0,N0), slice(0,N1), slice(N2-dU2,N2), slice(N3-dU3,N3)] ]

    -------
    """

    assert len(shape) == len(chunk), "Size of shape must match chunk"
    lenchunk = len(chunk)
    slicing = [[slice(j, j+chunk[i]) for j in range(0, shape[i], chunk[i])] for i in range(lenchunk)]

    iteration = product(*slicing)
    if tolist:
        iteration = list(iteration)

    return iteration

def contract_slicer(shape:tuple, chunk:None|tuple, comm:MPI.Intercomm):
    assert len(shape) == len(chunk), "Size of shape must match chunk"
    lenchunk = len(chunk)
    slicing = [[slice(j, j+chunk[i]) for j in range(0, shape[i], chunk[i])] for i in range(lenchunk)]

    iteration = product(*slicing)

    return iteration

def contract_slice(shape:tuple, chunk:None|tuple, comm:MPI.Intercomm, gather_to_rank0=True):
    """
    Slice the contraction legs.

    Suppose we calculate the tensor contraction M_{a,b} = Σ_{j,k,l} A_{a,j,k,l} * B_{b,j,k,l} ,
    and want to slice the legs `k` and `l` into k/dk and l/dl parts, in which the range of legs 
    are `a`∈[0, A-1], `b`∈[0, B-1], `j`∈[0, J-1], `k`∈[0, K-1] and `l`∈[0, L-1]. 
    In this case, contraction becomes M_{a,b} = Σ_{j,ik,il} Σ_{j,k',l'} A_{a,j,k',l'} * B_{b,j,k',l'}, 
    in which ik∈[0, nK-1], nK=K/dK, k'∈[ik*dK, (ik+1)*dK-1], il∈[0, nL-1], nL=L/dL, l'∈[il*dL, (il+1)*dL-1].
    In this case, `shape`=(J, K, L), `chunk`=(J, dK, dL)

    Parameters
    ----------
    shape : Shape of legs which are contracted.

    chunk : Devide the contracted legs into given chunk. 

    gather_to_rank0 : If True, gather the slices to rank 0.

    ----------

    Retruns
    ----------
    If gather_to_rank0 == True, return list of all combination of slices at rank0 and None at other ranks.
    If gather_to_rank0 == False, keeps the combination of slices at correspond rank.

    ----------

    Example
    ----------
    In the case of given example, the list have nK*nL elements. 
    Each element includes 3 slices for legs J, K and L respectively.
    For example:
        input: `shape`=(J, K, L), `chunk`=(J, dK, dL)
        return: `slice_combination`
    The n-th element of `slice_combination[n]` is [`slice(0,J)`, `slice(ik*dK, (ik+1)*dK)`, `slice(il*dl, (il+1)*dl)`]

    ----------
    """

    MPI_COMM = comm
    MPI_SIZE = MPI_COMM.Get_size()
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_INFO = MPI_COMM.Get_info()

    if chunk is not None:
        assert len(chunk) == len(shape), "dimension of chunk should match shape"
    else:
        chunk = tuple([1 for _ in range(len(shape))])

    slicing = [ [slice(j, min(j+chunk[i], shape[i])) for j in range(0, shape[i], chunk[i])] 
                 for i in range(len(chunk)) ]
    MPI_COMM.barrier()

    slice_list = []
    for n, iteration in enumerate(product(*slicing)):
        if n % MPI_SIZE == MPI_RANK:
            slice_list.append(iteration)
    MPI_COMM.barrier()

    if gather_to_rank0:
        slice_list = MPI_COMM.gather(slice_list, root=0)
        if MPI_RANK == 0:
            slice_list = chain.from_iterable(slice_list)
        MPI_COMM.barrier()
    
    return slice_list

def mapping_infomation(shape:tuple, chunk:tuple, comm:MPI.Intercomm):
    """
    return slices on each rank
    """
    lenchunk = len(chunk)
    assert lenchunk == len(shape), "dimension of chunk should match shape"

    slices = [[slice(j, min(j+chunk[i], shape[i])) for j in range(0, shape[i], chunk[i])] for i in range(lenchunk)]
    slices = product(*slices)

    size = comm.Get_size()
    local_map_info = [[] for _ in range(size)]
    for i, s in enumerate(slices):
        rank = i % size
        local_map_info[rank].append(s)

    return local_map_info

def flatten_2dim_job_results(A:list, job_size:int, comm:MPI.Intercomm):
    WORLD_MPI_SIZE = comm.Get_size()
    to_1dim = []
    for i in range(math.ceil(job_size / WORLD_MPI_SIZE)):
        for j in range(WORLD_MPI_SIZE):
            if i * WORLD_MPI_SIZE + j < job_size:
                to_1dim.append(A[j][i])
    return to_1dim

def contract_mpi(subscripts:str, operands:list, shapes:list, chunks=list, optimize=None|str, use_gpu=False):
    """
    Parameters
    ----------
    subscripts:

    *operands:

    shape:

    chunk:

    optimize:

    ----------

    Retruns
    ----------

    ......

    ----------
    """

    n_operands = len(operands)
    assert len(shapes) == len(chunks) and len(shapes) == n_operands and len(shapes) == len(chunks)
    
    local_sum = 0




#def contraction_slicer(shape:tuple, chunk=None|tuple):
#    if chunk is not None:
#        assert len(chunk) == len(shape), "dimension of chunk should match leg_shape"
#    else:
#        chunk = tuple([1 for _ in range(len(shape))])
#
#    slicing = [ [slice(j, min(j+chunk[i], shape[i])) for j in range(0, shape[i], chunk[i])] 
#                 for i in range(len(chunk)) ]
#    return slicing
#
#def slice_list_distributer(slicing):
#    slice_list = []
#    for n, iteration in enumerate(product(*slicing)):
#        if n % MPI_SIZE == MPI_RANK:
#            slice_list.append(iteration)
#    return slice_list
#
#def gather_slice_list(slice_list):
#    _slice_list = MPI_COMM.gather(slice_list, root=0)
#    if MPI_RANK == 0:
#        _slice_list = chain(_slice_list)
#    return _slice_list
