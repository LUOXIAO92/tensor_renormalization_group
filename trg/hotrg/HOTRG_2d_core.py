import os
import numpy as np
#import cupy as cp
import sys
import time
import math

from mpi4py import MPI

import opt_einsum as oe
from itertools import product

from .HOTRG import HOTRG_info as Info
from .HOTRG import Tensor_HOTRG as Tensor
from tools.linalg_tools import svd, eigh
from tools.mpi_tools import contract_slice, gpu_syn



def tranpose(where, T, do_what:str, direction:str, comm:MPI.Intercomm):
    """
    Parameters
    ----------
    `T` : Tensor T_{xXyY}
    >>>      Y
    >>>      |
    >>>  x---T---X
    >>>      |
    >>>      y

    `do_what` : "transpose" or "restore"
    `direction` : "x" or "X", x direction; "y" or "Y", y direction. 

    ----------
    """

    if comm.Get_rank() == where:
        if do_what == "transpose":
            if direction == "Y" or direction == "y":
                pass
            elif direction == "X" or direction == "x":
                T = T.transpose(axes=(2,3,0,1))

        elif do_what == "restore":
            if direction == "Y" or direction == "y":
                pass
            elif direction == "X" or direction == "x":
                T = T.transpose(axes=(2,3,0,1))

    comm.barrier()
    return T

def squeezer(where, T0, T1, T2, T3, 
             Dcut:int, 
             nrgsteps:int,
             comm:MPI.Intercomm, 
             chunk        = None|tuple, 
             truncate_eps = 0, 
             xp           = np,
             usegpu       = False,
             verbose      = False, 
             save_details = False,
             outdir       = None|str):
    """
    >>>     d        f                d                f
    >>>     |    j   |                |                |
    >>> c---T1-------T2---e       c---T1---\\       /---T2---e
    >>>     |        |                |     \\     /    |
    >>>    i|        |k              i|    P1\\---/P2   |k
    >>>     |        |                |     /     \\    |
    >>> a---T0-------T3---g       a---T0---/       \\---T3---g
    >>>     |    l   |                |                |
    >>>     b        h                b                h 
    
    #returns left/down projector PLD and right/up projector PRU 
    #PLD_{x,x0,x1} or PLD_{y,y0,y1} 
    #PRU_{x'0,x'1,x'} or PRU_{y'0,y'1,y'}
    
    """
    
    MPI_SIZE = comm.Get_size()
    MPI_RANK = comm.Get_rank()

    if MPI_RANK == where:
        assert (T0 is not None) and (T1 is not None) and (T2 is not None) and (T3 is not None), 'All Tensors must be kept on the same rank'
    
    def cal_R(truncate_eps=0, qr='qr', return_eigval=False):
        if qr == 'qr':
            if MPI_RANK == where:
                contract_shape = T0.shape[0], T0.shape[1], T1.shape[0], T1.shape[3], T0.shape[3], T0.shape[3]
                M_shape        = T0.shape[1], T1.shape[1], T0.shape[1], T1.shape[1]
            else:
                contract_shape = None
                M_shape        = None
            contract_shape = comm.bcast(obj=contract_shape, root=where)
            M_shape        = comm.bcast(obj=M_shape       , root=where)
            
            contract_chunk = chunk[0], chunk[1], chunk[0], chunk[1], chunk[2], chunk[3]
            contract_slices = contract_slice(shape=contract_shape, chunk=contract_chunk, comm=comm, gather_to_rank0=True)
            
            #M = oe.contract("aibe,akbf,cjed,clfd->ijkl", xp.conj(T0), T0, xp.conj(T1), T1)
            #if MPI_RANK == where:
            #    operands = [[] for _ in range(MPI_SIZE)]
            #    for n, legs in enumerate(contract_slices):
            #        a, b, c, d, e, f = legs
            #        operands[n % MPI_SIZE].append([xp.conj(T0[a,:,b,e]), T0[a,:,b,f], xp.conj(T1[c,:,e,d]), T1[c,:,f,d]])
            #else:
            #    operands = None
            #operands = comm.scatter(sendobj=operands, root=where)
            
            contract_iter = 
            for n, legs in enumerate(contract_slices):
                
        elif qr == 'rq':
            if MPI_RANK == where:
                contract_shape = T3.shape[1], T3.shape[2], T2.shape[1], T2.shape[3], T3.shape[3], T3.shape[3]
                M_shape        = T3.shape[0], T2.shape[0], T3.shape[0], T2.shape[0]
            else:
                contract_shape = None
                M_shape        = None
            contract_shape = comm.bcast(obj=contract_shape, root=where)
            M_shape        = comm.bcast(obj=M_shape       , root=where)

            contract_chunk = chunk[0], chunk[1], chunk[0], chunk[1], chunk[2], chunk[3]
            contract_slices = contract_slice(shape=contract_shape, chunk=contract_chunk, comm=comm, gather_to_rank0=True)
            
            #M = oe.contract("iabe,kabf,jced,lcfd->ijkl", T3, xp.conj(T3), T2, xp.conj(T2))
            if MPI_RANK == where:
                operands = [[] for _ in range(MPI_SIZE)]
                for n, legs in enumerate(contract_slices):
                    a, b, c, d, e, f = legs
                    operands[n % MPI_SIZE].append([T3[:,a,b,e], xp.conj(T3[:,a,b,f]), T2[:,c,e,d], xp.conj(T2[:,c,f,d])])
            else:
                operands = None
            operands = comm.scatter(sendobj=operands, root=where)

        subscripts = {'qr' : 'aibe,akbf,cjed,clfd->ijkl',
                      'rq' : 'iabe,kabf,jced,lcfd->ijkl'}
        local_M = xp.zeros(shape=M_shape, dtype=complex)
        gpu_syn(usegpu)
        comm.barrier()
        t0  = time.time()
        t00 = time.time()

        for n, ops in enumerate(operands):
            local_M += oe.contract(subscripts[qr], *ops)
            if verbose:
                if (MPI_RANK == 0) and (n % 100 == 0) and (n > 0):
                    t1 = time.time()
                    print(f"Local iteration:{n} at rank{MPI_RANK}, time= {t1-t0:.2e} s")
                    t0 = time.time()
                elif (MPI_RANK == 0) and (n == 0):
                    print(f"Start calculate reduced density matrix.")
        
        M = comm.reduce(sendobj=local_M, op=MPI.SUM, root=0)
        gpu_syn(usegpu)
        comm.barrier()
        t11 = time.time()
        if MPI_RANK == 0:
            print(f"Reduced density matrix calculation finished, time= {t11-t00:.2e} s")

        
        if MPI_RANK == where:
            M = xp.reshape(M,  (M.shape[0]*M.shape[1], M.shape[2]*M.shape[3]))
            Eigvect, Eigval, _ = svd(M, shape=[[0], [1]], k=min(*M.shape), truncate_eps=truncate_eps)
            if qr == 'qr':
                R = oe.contract("ia,a->ai", xp.conj(Eigvect), xp.sqrt(Eigval))
            elif qr == 'rq':
                R = oe.contract("ia,a->ia", Eigvect, xp.sqrt(Eigval))
        else:
            R = None
            Eigval = None
        gpu_syn(usegpu)
        comm.barrier()

        if return_eigval:
            return R, Eigval
        else:
            return R
    
    R1, Eigval1 = cal_R(truncate_eps, qr='qr', return_eigval=True)
    R2, Eigval2 = cal_R(truncate_eps, qr='rq', return_eigval=True)
    gpu_syn(usegpu)
    comm.barrier()
    
    if MPI_RANK == 0:
        #U, S, VH = cp.linalg.svd(R1@R2)
        R1R2 = R1@R2
        k = min(*R1R2.shape, Dcut)
        U, S, VH = svd(R1R2, shape=[[0], [1]], k=k, truncate_eps=truncate_eps)
        UH = xp.conj(U.T)
        Sinv = 1 / S
        #Sinv = Sinv.astype(cp.complex128)
        V = xp.conj(VH.T)

        print("eL",Eigval1[:k])
        print("eR",Eigval2[:k])
        print("S", S[:k])

        del U, S, VH

        P1 = oe.contract("ia,aj,j->ij", R2, V , xp.sqrt(Sinv))
        P2 = oe.contract("i,ia,aj->ij", xp.sqrt(Sinv), UH, R1)
        
        print("Tr(P1@P2)=", xp.trace(P1@P2), "|P1@P2|^2=", xp.linalg.norm(P1@P2)**2)
    
        P1_shape = (T0.shape[1], T1.shape[1], P1.shape[1])
        P2_shape = (P2.shape[0], T3.shape[0], T2.shape[0])
        P1 = xp.reshape(P1, P1_shape)
        P2 = xp.reshape(P2, P2_shape)

        if save_details:
            output_dir = outdir.rstrip('/') + '/squeezer'
            filename = output_dir + f'/squeezer_n{nrgsteps}.dat'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            if nrgsteps == 0:
                mode = 'w'
            else:
                mode = 'a'
            
            with open(filename, mode) as out:
                E10, E20 = xp.max(Eigval1), xp.max(Eigval2)
                E1 = Eigval1 / E10
                E2 = Eigval2 / E20
                out.write(f'#λ_l={E10:.12e}, λ_r={E20:.12e}\n')
                for e1, e2 in zip(E1, E2):
                    out.write(f'{e1:.12e} {e2:.12e}\n')

    else:
        P1 = None
        P2 = None
    del R1, R2, Eigval1, Eigval2
    gpu_syn(usegpu)
    comm.barrier()

    return P2, P1

def coarse_graining(T0, T1, PLD, PRU, xp, comm:MPI.Intercomm, chunk=None|tuple, usegpu=False, verbose=False):
    """
    >>> T0_{acke}, T1_{bedl}, PLD_{iab}, PRU_{cdj}
    >>>            l
    >>>            |
    >>>       /b---T1---d\\
    >>> i--PLD     |      d
    >>>      \\     e--e    \\
    >>>       a       |     PRU--j
    >>>         \\a---T0---c/  
    >>>               |   
    >>>               k
    
    Parameters
    ----------
    `T0`, `T1` : Tensors
    `PLD`, `PRU` : Squeezers
    `comm` : MPI.Intercomm
    `chunk`: Chunk of contraction. `chunk`=(chunk_a, chunk_d, chunk_e). 

    ----------
    """
    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()

    χs = None
    if MPI_RANK == 0:
        dtype = T0.dtype
        χ_a = PLD.shape[1]
        χ_e = T0.shape[2]
        χ_d = T1.shape[0]
        χ_i = PLD.shape[0]
        χ_j = PRU.shape[2]
        χ_k = T0.shape[2]
        χ_l = T1.shape[3]
        χs = [χ_a, χ_e, χ_d, χ_i, χ_j, χ_k, χ_l, dtype]
    χs = comm.scatter(sendobj=χs, root=0)
    χ_a, χ_e, χ_d, χ_i, χ_j, χ_k, χ_l, dtype = χs
    del χs
    
    contract_slices = contract_slice(shape=(χ_a, χ_d, χ_e), chunk=chunk, comm=comm, gather_to_rank0=True)
    if MPI_RANK == 0:
        operands = [[] for _ in range(MPI_SIZE)]
        for n, legs in enumerate(contract_slices):
            a, d, e = legs
            if n % MPI_SIZE == MPI_RANK:
                operands[MPI_RANK].append([T0[a,:,:,e], T1[:,d,e,:], PLD[:,a,:], PRU[:,d,:]])
    else:
        operands = None
    operands = comm.scatter(sendobj=operands, root=0)
    gpu_syn(usegpu)
    comm.barrier()
    #del T0, T1, PLD, PRU

    path = [(0, 2), (0, 1), (0, 1)]
    subscripts = "acke,bdel,iab,cdj->ijkl"
    local_T = xp.zeros(shape=(χ_i, χ_j, χ_k, χ_l), dtype=dtype)

    gpu_syn(usegpu)
    comm.barrier()
    t0  = time.time()
    t00 = time.time()

    for n, ops in operands:
        local_T += oe.contract(subscripts, *ops, optimize=path)

        if verbose:
            if (MPI_RANK == 0) and (n % 100 == 0) and (n > 0):
                t1 = time.time()
                print(f"Local iteration:{n} at rank{MPI_RANK}, time= {t1-t0:.2e} s")
                t0 = time.time()
            elif (MPI_RANK == 0) and (n == 0):
                print(f"Start coarse graining.")
    
    T = comm.reduce(sendobj=local_T, op=MPI.SUM, root=0)
    gpu_syn(usegpu)
    comm.barrier()
    t11 = time.time()
    if MPI_RANK == 0:
        print(f"Coarse graining calculation finished, time= {t11-t00:.2e} s")

    return T

#def normalize(T, usegpu=False):
#    if usegpu:
#        from cupy import abs
#    else:
#        from numpy import abs
#    c = oe.contract("iijj", T)
#    c = abs(c)
#    T = T / c
#    return T, c

def new_pure_tensor(info:Info,
                    T:Tensor, 
                    direction:str):
    
    where    = T.where
    usegpu   = T.usegpu
    nrgsteps = T.nrgsteps
    Dcut     = T.Dcut
    xp       = T.xp

    truncate_eps = info.truncate_eps
    gilt_eps     = info.gilt_eps 
    Ngilt        = info.Ngilt    
    gilt_legs    = info.Ncutlegs
    reduced_matrix_chunk  = info.reduced_matrix_chunk 
    coarse_graining_chunk = info.coarse_graining_chunk
    
    verbose      = info.verbose    
    save_details = info.save_details
    outdir       = info.outdir     
    
    comm = info.comm
    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()

    gpu_syn(T.usegpu)
    comm.barrier()
    if Ngilt == 1:
        if direction == 'y' or direction == 'Y':
            T0, T1 = gilt_plaq(where, T.T, T.T, comm, gilt_eps, direction, gilt_legs, T.usegpu)
        elif direction == 'x' or direction == 'X':
            T0, T1 = T.T, T.T
    elif Ngilt == 2:
        T0, T1 = gilt_plaq(where, T.T, T.T, comm, gilt_eps, direction, gilt_legs, T.usegpu)
    gpu_syn(T.usegpu)
    comm.barrier()

    T0 = tranpose(where, T0, 'transpose', direction, comm)
    T1 = tranpose(where, T1, 'transpose', direction, comm)
    gpu_syn(usegpu)
    comm.barrier()

    PLD, PRU = squeezer(where, T0, T1, T1, T0, Dcut, nrgsteps, 
                        comm, 
                        reduced_matrix_chunk, 
                        truncate_eps, 
                        xp,
                        usegpu, 
                        verbose, 
                        save_details, 
                        outdir)
    gpu_syn(usegpu)
    comm.barrier()

    t0 = time.time()
    T.T = coarse_graining(T0, T1, PLD, PRU, xp, comm, coarse_graining_chunk, usegpu, verbose)
    gpu_syn(usegpu)
    comm.barrier()
    t1 = time.time()

    del PLD, PRU, T0, T1

    T.T = tranpose(where, T.T, "restore", direction, comm)
    gpu_syn(usegpu)
    comm.barrier()
    
    return T

def new_impuer_tensor_2imp(T, Timp0, Timp1, nx, ny, Dcut:int, direction:str, comm:MPI.Intercomm, usegpu=False):
    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()
    
    if MPI_RANK == 0:
        T     = tranpose(T,     direction)
        Timp0 = tranpose(Timp0, direction)
        Timp1 = tranpose(Timp1, direction)
    gpu_syn(usegpu)
    comm.barrier()

    PLD, PRU = squeezer(T, T, T, T, Dcut)
    gpu_syn(usegpu)
    comm.barrier()

    t0 = time.time()
    #Timp0 = oe.contract("acke,bdel,iab,cdj->ijkl", Timp0, T, PLD, PRU)
    Timp0 = coarse_graining(Timp0, T, PLD, PRU, comm, chunk=(T.shape[0], 1, 1), usegpu=usegpu)
    gpu_syn(usegpu)
    comm.barrier()
    
    if (ny%2 == 1) and (direction == "Y" or direction == "y"):
        Timp1 = coarse_graining(Timp1, T, PLD, PRU, comm, chunk=(T.shape[0], 1, 1), usegpu=usegpu)

    if (ny%2 == 0) and (direction == "Y" or direction == "y"):
        Timp1 = coarse_graining(Timp1, T, PLD, PRU, comm, chunk=(T.shape[0], 1, 1), usegpu=usegpu)

    if (nx%2 == 1) and (direction == "X" or direction == "x"):
        Timp1 = coarse_graining(T, Timp1, PLD, PRU, comm, chunk=(T.shape[0], 1, 1), usegpu=usegpu)

    if (nx%2 == 0) and (direction == "X" or direction == "x"):
        Timp1 = coarse_graining(Timp1, T, PLD, PRU, comm, chunk=(T.shape[0], 1, 1), usegpu=usegpu)
    gpu_syn(usegpu)
    comm.barrier()

    T = coarse_graining(T, T, PLD, PRU, comm, chunk=(T.shape[0], 1, 1), usegpu=usegpu)
    gpu_syn(usegpu)
    comm.barrier()
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/3))

    if MPI_RANK == 0:
        T     = tranpose(T    , direction, "restore")
        Timp0 = tranpose(Timp0, direction, "restore")
        Timp1 = tranpose(Timp1, direction, "restore")
    gpu_syn(usegpu)
    comm.barrier()

    del PLD, PRU

    return T, Timp0, Timp1

def new_impuer_tensor_2to1imp(T, Timp0, Timp1, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = tranpose(T, direction)
    Timp0 = tranpose(Timp0, direction)
    Timp1 = tranpose(Timp1, direction)
    PLD, PRU = squeezer(T, T, T, T, Dcut)

    t0 = time.time()
    Timp0 = oe.contract("acke,bdel,iab,cdj->ijkl", Timp0, Timp1, PLD, PRU)
    T = oe.contract("acke,bdel,iab,cdj->ijkl", T, T, PLD, PRU)
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/2))

    del Timp1, PLD, PRU

    T     = tranpose(T    , direction)
    Timp0 = tranpose(Timp0, direction)

    return T, Timp0

def new_impuer_tensor_1imp(T, Timp0, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = tranpose(T    , direction)
    Timp0 = tranpose(Timp0, direction)
    PLD, PRU = squeezer(T, T, T, T, Dcut)

    Timp0 = oe.contract("acke,bdel,iab,cdj->ijkl", Timp0, T, PLD, PRU)
    T = oe.contract("acke,bdel,iab,cdj->ijkl", T, T, PLD, PRU)

    t0 = time.time()
    T     = tranpose(T    , direction)
    Timp0 = tranpose(Timp0, direction)
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/2))

    del PLD, PRU

    return T, Timp0

#hotrg routine end------------------------------------------------------------------

#gilt hotrg start-------------------------------------------------------------------------
convergence_err = 1e-2
max_iteration = 1000

def optimize_Rp(U, S, gilt_eps, usegpu=False):
    """
    U: U_{αβ,i}
    """
    if usegpu:
        import cupy as xp
    else:
        import numpy as xp

    #compute initial Rp
    Rp, Rp_ai, s, Rp_ib = compute_Rp(U, S, gilt_eps, need_svd=True, split=True, usegpu=usegpu)
    U_inner = U
    S_inner = S
    us_inner = Rp_ai
    svh_inner = Rp_ib

    time_diff = 0
    count = 0
    global convergence_err
    #compute and truncate bond matrix untill all singular values converge to 1
    while (xp.abs(s-1).max() >= convergence_err) and (count < max_iteration):
        t0 = time.time()
        #compute inner enviroment tensor and then svd it
        E_inner = oe.contract("ABi,Aa,bB,i->abi", U_inner, us_inner, svh_inner, S_inner)
        U_inner, S_inner, _ = svd(E_inner, shape=[[0,1], [2]])
        S_inner = S_inner / xp.sum(S_inner)
        

        #compute truncated bond matrix and split it into two parts
        _, us_inner, s, svh_inner = compute_Rp(U_inner, S_inner, gilt_eps, need_svd=True, split=True, usegpu=usegpu)
        Rp_ai = oe.contract("aA,Ai->ai", Rp_ai, us_inner)
        Rp_ib = oe.contract("iB,Bb->ib", svh_inner, Rp_ib)
        count += 1
        t1 = time.time()
        time_diff += t1-t0
        if count % 20 == 0:
            #__sparse_check__(E_inner, "inner Env tensor")
            print("iteration:{}, s[:20]:{}, time:{:.2e}s".format(count, s[:20], time_diff))
            time_diff = 0

        del E_inner

    Rp = oe.contract("ai,ib->ab", Rp_ai, Rp_ib)
    del U_inner, S_inner, us_inner, svh_inner, s

    return Rp, count

def compute_Rp(U, S, gilt_eps, need_svd:bool, split=False, usegpu=False):
    if usegpu:
        import cupy as xp
    else:
        import numpy as xp

    #compute trace t_i=TrU_i
    t = oe.contract("aai->i", U)

    #compute t'_i = t_i * S_i^2 / (ε^2 + S_i^2)
    if gilt_eps != 0:
        ratio = S / gilt_eps
        weight = ratio**2 / (1 + ratio**2)
        tp = t * weight
    else:
        tp = t

    #compute R'
    Rp = oe.contract("i,abi->ab", tp, xp.conj(U))

    del tp

    #svd R' 
    if need_svd or split:
        u, s, vh = svd(Rp, shape=[[0], [1]], truncate_err=gilt_eps*1e-3, split=split)
        return Rp, u, s, vh
    else:
        return Rp
    
def gilt_plaq_routine(T0, T1, gilt_eps, leg:str, direction:str, usegpu=False):
    """
    >>>       direction:Y            direction:X      
    >>>       |        |             |        |     
    >>>    ---T0---j---T0---      ---T0---i---T1--- 
    >>>       |        |             |        |     
    >>>       i        k             l        j     
    >>>       |        |             |        |     
    >>>    ---T1---l---T1---      ---T0---k---T1--- 
    >>>       |        |             |        |     
    gilting leg = i,j,k,l
    with tensors T_{x,x',y,y'}
    """
    if usegpu:
        import cupy as xp
    else:
        import numpy as xp

    #compute eigenvalues of environment tensor, S:=S^2
    #     -------
    #     |     |    
    #  |--LU----RU--|
    #  |  |     |   |
    #  |  |     |   |
    #  |--LD----RD--|
    #     |     |    
    #     -------
    if direction == "y" or direction == "Y":
        LD = oe.contract("ixjy,iXjY->xyXY", T1, xp.conj(T1))
        LU = oe.contract("ixyj,iXYj->xyXY", T0, xp.conj(T0))
        RU = oe.contract("xiyj,XiYj->xyXY", T0, xp.conj(T0))
        RD = oe.contract("xijy,XijY->xyXY", T1, xp.conj(T1))
    elif direction == "x" or direction == "X":
        LD = oe.contract("ixjy,iXjY->xyXY", T0, xp.conj(T0))
        LU = oe.contract("ixyj,iXYj->xyXY", T0, xp.conj(T0))
        RU = oe.contract("xiyj,XiYj->xyXY", T1, xp.conj(T1))
        RD = oe.contract("xijy,XijY->xyXY", T1, xp.conj(T1))

    if direction == "y" or direction == "Y": 
        if   leg == "i":
            subscripts = "aαbA,cβdB,cedf,aebf->αβAB"
        elif leg == "j":
            subscripts = "acbd,αcAd,βeBf,aebf->αβAB"
        elif leg == "k":
            subscripts = "acbd,ecfd,eαfA,aβbB->αβAB"
        elif leg == "l":
            subscripts = "αaAb,cadb,cedf,βeBf->αβAB"
    elif direction == "x" or direction == "X": 
        if   leg == "l":
            subscripts = "aαbA,cβdB,cedf,aebf->αβAB"
        elif leg == "i":
            subscripts = "acbd,αcAd,βeBf,aebf->αβAB"
        elif leg == "j":
            subscripts = "acbd,ecfd,eαfA,aβbB->αβAB"
        elif leg == "k":
            subscripts = "αaAb,cadb,cedf,βeBf->αβAB"
    
    Env = oe.contract(subscripts, LD, LU, RU, RD)
    S, U = eigh(Env, shape=[[0,1], [2,3]])
    S[S < 0] = 0
    S = xp.sqrt(S)
    S = S / xp.sum(S)
    del LD, LU, RU, RD, Env

    Rp, count = optimize_Rp(U, S, gilt_eps)
    uRp, sRp, vRph = svd(Rp, shape=[[0], [1]], truncate_err=gilt_eps*1e-3)
    global convergence_err
    done = xp.abs(sRp-1).max() < convergence_err
    Rp_ai = oe.contract("ai,i->ai", uRp,  xp.sqrt(sRp))
    Rp_ib = oe.contract("ib,i->ib", vRph, xp.sqrt(sRp))
    print("sRp:", sRp)
    err = gilt_error(U, S, Rp_ai, Rp_ib)

    if gilt_eps != 0:
        if direction == "y" or direction == "Y": 
            if leg == "i":
                T1 = oe.contract("xXya,ai->xXyi", T1, Rp_ai)
                T0 = oe.contract("xXbY,ib->xXiY", T0, Rp_ib)
            elif leg == "j":
                T0 = oe.contract("xayY,ai->xiyY", T0, Rp_ai) 
                T0 = oe.contract("bXyY,ib->iXyY", T0, Rp_ib)
            elif leg == "k":
                T0 = oe.contract("xXaY,ai->xXiY", T0, Rp_ai)
                T1 = oe.contract("xXyb,ib->xXyi", T1, Rp_ib)
            elif leg == "l":
                T1 = oe.contract("xayY,ai->xiyY", T1, Rp_ai) 
                T1 = oe.contract("bXyY,ib->iXyY", T1, Rp_ib)
        elif direction == "x" or direction == "X": 
            if leg == "i":
                T0 = oe.contract("xayY,ai->xiyY", T0, Rp_ai)
                T1 = oe.contract("bXyY,ib->iXyY", T1, Rp_ib)
            elif leg == "j":
                T1 = oe.contract("xXaY,ai->xXiY", T1, Rp_ai)
                T1 = oe.contract("xXyb,ib->xXyi", T1, Rp_ib)
            elif leg == "k":
                T0 = oe.contract("xayY,ai->xiyY", T0, Rp_ai)
                T1 = oe.contract("bXyY,ib->iXyY", T1, Rp_ib)
            elif leg == "l":
                T0 = oe.contract("xXya,ai->xXyi", T0, Rp_ai) 
                T0 = oe.contract("xXbY,ib->xXiY", T0, Rp_ib)
    
    return T0, T1, err, done, count
    

def gilt_error(U, S, Rp_ai, Rp_ib, usegpu=False):
    if usegpu:
        import cupy as xp
    else:
        import numpy as xp

    t = xp.einsum("iij->j", U)
    tp = xp.einsum("abt,ai,ib->t", U, Rp_ai, Rp_ib)
    diff = t-tp
    diff = diff*S
    err = xp.linalg.norm(diff) / xp.linalg.norm(t*S)
    return err

def gilt_plaq(where, T0, T1, 
              comm:MPI.Intercomm, 
              gilt_eps  = 1e-7, 
              direction = "y", 
              gilt_legs = 2,
              usegpu    = False):
    
    if gilt_eps < 1e-12:
        return T0, T1

    gilt_err = 0
    from itertools import cycle
    if gilt_legs == 2:
        legs = 'jl'
    elif gilt_legs == 4:
        legs = 'ijkl'

    if comm.Get_rank() == where:
        done_legs = {leg:False for leg in legs}
        for leg in cycle(legs):
            T0_shape0, T1_shape0 = T0.shape, T1.shape
            t0 = time.time()
            T0.T, T1.T, err, done, count = gilt_plaq_routine(T0, T1, gilt_eps=gilt_eps, leg=leg, direction=direction, usegpu=usegpu)
            t1 = time.time()
            T0_shape1, T1_shape1 = T0.shape, T1.shape
            gilt_err += err
            print("T0:{}->{}\nT1:{}->{}\nleg:{}, gilt err= {:.6e}, iteration count:{:}, done:{}, time:{:.2e}s"\
                  .format(T0_shape0, T0_shape1, T1_shape0, T1_shape1, leg, gilt_err, count, done, t1-t0))
            done_legs[leg] = True

            if all(done_legs.values()):
                break
    gpu_syn(usegpu)
    comm.barrier()

    return T0, T1

#gilt hotrg end---------------------------------------------------------------------------