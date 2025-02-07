
import opt_einsum as oe
import numpy as np
import time
import math
from itertools import product
#import cupy as cp

from mpi4py import MPI
#WORLD_MPI_COMM = MPI.COMM_WORLD
#node = MPI.Get_processor_name()

from tools.mpi_tools import gpu_syn, flatten_2dim_job_results, pure_gauge_slice

def SU2_matrix_Gauss_Legendre_quadrature(Kθ:int, Kα:int, Kβ:int, comm:MPI.Intercomm, to_cupy=False):
    WORLD_MPI_COMM = comm
    WORLD_MPI_RANK = WORLD_MPI_COMM.Get_rank()
    WORLD_MPI_SIZE = WORLD_MPI_COMM.Get_size()
    info = WORLD_MPI_COMM.Get_info()
    
    from scipy.special import roots_legendre
    θ, wt = roots_legendre(Kθ)
    α, wa = roots_legendre(Kα)
    β, wb = roots_legendre(Kβ)
    θ = np.asarray(np.pi * (θ + 1) / 4)
    α = np.asarray(np.pi * (α + 1))
    β = np.asarray(np.pi * (β + 1))

    epia = np.exp( 1j*α)
    epib = np.exp( 1j*β)
    emia = np.exp(-1j*α)
    emib = np.exp(-1j*β)
    st = np.sin(θ)
    ct = np.cos(θ)
    It = np.ones(shape=Kθ, dtype=complex)
    Ia = np.ones(shape=Kα, dtype=complex)
    Ib = np.ones(shape=Kβ, dtype=complex)

    #Uij = Uij(θ, α, β)
    U = np.zeros(shape=(2, 2, Kθ, Kα, Kβ), dtype=complex)
    subscript = "θ,α,β->θαβ"
    U[0,0] =  oe.contract(subscript, ct, epia, Ib)
    U[0,1] =  oe.contract(subscript, st, Ia, epib)
    U[1,0] = -oe.contract(subscript, st, Ia, emib)
    U[1,1] =  oe.contract(subscript, ct, emia, Ib)
    
    I = np.zeros_like(U)
    I[0,0] = oe.contract(subscript, It, Ia, Ib)
    I[1,1] = oe.contract(subscript, It, Ia, Ib)

    #w[0] = contract("α,α,α->α", cp.sin(theta), cp.cos(theta), w[0])
    #Jacobian = (π/8) * sin(θ)cos(θ)
    Jt = st*ct
    Ja = Ia
    Jb = Ib
    J = oe.contract(subscript, Jt, Ja, Jb) * (np.pi / 8)
    w = oe.contract(subscript, wt, wa, wb)
    #weight

    U = np.reshape(U, shape=(2, 2, Kθ * Kα * Kβ))
    I = np.reshape(I, shape=(2, 2, Kθ * Kα * Kβ))
    w = np.reshape(w, shape=(Kθ * Kα * Kβ))
    J = np.reshape(J, shape=(Kθ * Kα * Kβ))
    if to_cupy:
        import cupy as cp
        U = cp.asarray(U)
        I = cp.asarray(I)
        w = cp.asarray(w)
        J = cp.asarray(J)
    
    return U, w, J, I

#def norm_for_elements(x):
#    if type(x) == np.ndarray:
#        sqrt = np.sqrt
#    else:
#        import cupy as cp
#        sqrt = cp.sqrt
#
#    norm = sqrt((x.real)**2 + (x.imag)**2)
#    return norm

def admissibility_condition(TrP, ε:float):
    """
    Admissibility condition ||1 - U0† U1† U2 U3}|| < ε in Luscher's gauge action.

    Parameters
    ----------
    P : Gauge. numpy.ndarray or cupy.ndarray
    ε : Parameter of Luscher's gauge action

    ----------

    Retruns
    -------
    norm_{U0† U1† U2 U3} = ||1 - U0† U1† U2 U3}||, and bool index that satisfies ||1 - U0† U1† U2 U3}|| < ε

    -------
    """
    if type(TrP) == np.ndarray:
        conj = np.conj
        sqrt = np.sqrt
        abs  = np.abs
        linalgnorm = np.linalg.norm
        xp = np
    else:
        from cupy import conj, sqrt, abs
        from cupy.linalg import norm as linalgnorm
        import cupy as xp
    
    norm = 4 - 2 * TrP.real
    #norm = 2 - TrP.real
    assert linalgnorm(norm.imag) < 1e-12, "2 - 2*Re(P00) - 2*Re(P11) + Tr(PP†) must be real!"
    norm = sqrt(abs(norm))
    index = norm > ε
    return norm, index

def plaquette_contraction_for_hosvd(β:float, ε:float|None, U, w, J, leg_hosvd, iteration, 
                                    comm:MPI.Intercomm, use_gpu=False, verbose=False):
    WORLD_MPI_COMM = comm
    WORLD_MPI_RANK = WORLD_MPI_COMM.Get_rank()
    WORLD_MPI_SIZE = WORLD_MPI_COMM.Get_size()
    info = WORLD_MPI_COMM.Get_info()

    if use_gpu:
        from cupy import zeros, conj, exp, inf, sqrt
    else:
        from numpy import zeros, conj, exp, inf, sqrt
    
    N = U.shape[2]
    M_local = zeros(shape=(N, N), dtype=complex)

    subscripts = ["ijkl,Ijkl->iI", "ijkl,iJkl->jJ", "ijkl,ijKl->kK", "ijkl,ijkL->lL"]

    #WORLD_MPI_COMM.barrier()
    #if use_gpu:
    #    cp.cuda.get_current_stream().synchronize()
    t0 = time.time()
    t00 = time.time()
    for n, i in enumerate(iteration):
        i0, i1, i2, i3 = i
        if n % WORLD_MPI_SIZE == WORLD_MPI_RANK:

            #TrP = oe.contract("abi,bcj,dck,adl->ijkl", conj(U[:,:,i0]), conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
            TrP = oe.contract("dci,adj,abk,bcl->ijkl", conj(U[:,:,i0]), conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])

            if ε is not None:
                norm, idx = admissibility_condition(TrP, ε)
                A = (1 - 0.5*TrP.real) / (1 - (norm / ε))
                A = exp(-β * A)
                A[idx] = 0.0
            else:
                A = exp(-β * (1 - 0.5*TrP.real))

            A = oe.contract("i,j,k,l,i,j,k,l,ijkl->ijkl", sqrt(w[i0]), sqrt(w[i1]), sqrt(w[i2]), sqrt(w[i3]), 
                                                          sqrt(J[i0]), sqrt(J[i1]), sqrt(J[i2]), sqrt(J[i3]), A)
            M_local += oe.contract(subscripts[leg_hosvd], A, conj(A))
            num_inf = len(M_local[M_local == inf])
            assert num_inf == 0, f"Overflow at {n}th iteration. Have {num_inf} infs."

            if verbose:
                if (n > 0) and (n % (25*WORLD_MPI_SIZE) == 0) and (WORLD_MPI_RANK == 0):
                    t1 = time.time() if WORLD_MPI_RANK == 0 else None
                    print(f"Global iters:{n}. Local iters:{n // WORLD_MPI_SIZE}. {(t1-t0) / (n // WORLD_MPI_SIZE):.2e} sec/local_iter") #. Size of A: {A.nbytes/(1024**3):.2e} Gbytes")
                    t0 = time.time() if WORLD_MPI_RANK == 0 else None

    t11 = time.time()
    if WORLD_MPI_RANK == 0:
        print(f"Tot iteration {n+1}. Time= {t11-t00:.2e} s")

    gpu_syn(use_gpu)
    M = WORLD_MPI_COMM.reduce(sendobj=M_local, op=MPI.SUM, root=0)
    WORLD_MPI_COMM.barrier()

    return M


def env_tensor_for_3d_SU2_pure_gauge(A, direction:str, comm:MPI.Intercomm, use_gpu=False):
    WORLD_MPI_RANK = comm.Get_rank()
    WORLD_MPI_SIZE = comm.Get_size()

    if use_gpu:
        import cupy as xp
    else:
        xp = np

    if WORLD_MPI_RANK == 0:
        for i in range(A.ndim):
            for j in range(i, A.ndim):
                assert A.shape[i] == A.shape[j], "A must have the same bond dimension of 4 legs"
        
    #A = comm.bcast(obj=A, root=0)
    
    if WORLD_MPI_RANK == 0:
        Dcut = A.shape[0]
        I = xp.diag([1 for _ in range(Dcut)])
        B = oe.contract("Ωi,Ωj,Ωk,Ωl->ijkl", I, I, I, I)
    else:
        Dcut = None
        I, B = None, None
    del I
    Dcut = comm.bcast(obj=Dcut, root=0)


    def env_tensor_components(subscript_list:list, operand_list:list):
        
        results = []
        for job_id in range(4):

            #Map job_id-th job to dest_rank
            dest_rank  = job_id % WORLD_MPI_SIZE

            #Prepare the jobs on rank 0
            if WORLD_MPI_RANK == 0:
                subscripts = subscript_list[job_id]
                operands   = operand_list[job_id]
                sendjob = [subscripts, operands]
                
                if WORLD_MPI_RANK != dest_rank:
                    comm.send(sendjob=sendjob, dest=dest_rank, tag=dest_rank)
                else:
                    job = sendjob
                    
            #Recive job from rank 0
            else:
                job = comm.recv(source=0, tag=WORLD_MPI_RANK)

            if WORLD_MPI_RANK == dest_rank:
                subscripts, operands = job
                results.append(
                    oe.contract(subscripts, *operands)
                )

        gpu_syn(use_gpu)
        comm.barrier()

        results = comm.gather(sendobj=results, root=0)
        if WORLD_MPI_RANK == 0:
            Envs = flatten_2dim_job_results(results, job_size=4, comm=comm)
        else:
            Envs = None
        del results
        gpu_syn(use_gpu)
        comm.barrier()

        return Envs


    if direction == 'T' or direction == 't':
        #QR left----------------------------------------------------------------------------
        #Env_{T'02 T'12, T02 T12} = (QR)†QR = R†Q†QR = R†R = UΛU† → R = sqrt(Λ)U†)
        #job_list = [A0†A0, B01†B01, A1†A1, A2†A2]
        subscript_list = ["abgh,ABgh->abAB", "icaj,iCAj->caCA", "kdcl,kDCl->dcDC", "enof,EnoF->efEF"]

        if WORLD_MPI_RANK == 0:
            operand_list = [[xp.conj(A), A], [B, B], [xp.conj(A), A], [xp.conj(A), A]]
        else:
            operand_list = None
        AA, BB, CC, DD = env_tensor_components(subscript_list, operand_list)
        
        if WORLD_MPI_RANK == 0:
            C = DD
            D = oe.contract("abAB,caCA,dcDC->bBdD", AA, BB, CC)
        else:
            C, D = None, None
        del AA, BB, CC, DD, operand_list
        
        subscripts_left = "abAB,cCdD,biec,Bkec,fjad,flAD->ijkl"
        operands_left = [C, D, B, B, B, B]
        del B, C, D
        gpu_syn(use_gpu)
        comm.barrier()
        #QR left----------------------------------------------------------------------------
        
        #QR right---------------------------------------------------------------------------
        #Env_{d0 d1 d'0 d'1} = RQ(RQ)† = RQQ†R† = RR† = UΛU† → R = Usqrt(Λ)
        #job_list = [B01B01†, B12B12†, A2A2†, B02B02†]
        subscript_list = ["hcai,hCAi->caCA", "kled,klED->edED", "emnf,EmnF->efEF", "fopb,FopB->fbFB"]

        if WORLD_MPI_RANK == 0:
            operand_list = [[B, B], [B, B], [xp.conj(A), A], [B, B]]
        else:
            operand_list = None
        AA, BB, CC, DD = env_tensor_components(subscript_list, operand_list)

        if WORLD_MPI_RANK == 0:
            C = AA
            D = oe.contract("abAB,acAC,cdCD->bBdD", BB, CC, DD)
        else:
            C, D = None, None
        del AA, BB, CC, DD, operand_list

        subscript_right = "abAB,cCdD,bdei,BDek,fcaj,fCAl->ijkl"
        operands_right = [C, D, A, xp.conj(A), A, xp.conj(A)]
        del B, C, D
        gpu_syn(use_gpu)
        comm.barrier()
        #QR right---------------------------------------------------------------------------


    elif direction == 'X' or direction == 'x':
        #QR left----------------------------------------------------------------------------
        #Env_{T'02 T'12, T02 T12} = (QR)†QR = R†Q†QR = R†R = UΛU† → R = sqrt(Λ)U†)
        #job_list = [A0†A0, B01†B01, A1†A1, A2†A2]
        subscript_list = ["hcai,hCAi->caCA", "idck,iDCk->dcDC", "lmed,lmED->edED", "fopb,FopB->fbFB"]
    
        if WORLD_MPI_RANK == 0:
            operand_list = [[xp.conj(A), A], [xp.conj(B), B], [xp.conj(A), A], [xp.conj(A), A]]
        else:
            operand_list = None
        AA, BB, CC, DD = env_tensor_components(subscript_list, operand_list)
        
        if WORLD_MPI_RANK == 0:
            C = DD
            D = oe.contract("abAB,caCA,dcDC->bBdD", AA, BB, CC)
        else:
            C, D = None, None
        del AA, BB, CC, DD, operand_list
        
        subscripts_left = "abAB,cCdD,biec,Bkec,fjad,flAD->ijkl"
        operands_left = [C, D, B, B, B, B]
        del B, C, D
        gpu_syn(use_gpu)
        comm.barrier()
        #QR left----------------------------------------------------------------------------
        
        #QR right---------------------------------------------------------------------------
        #Env_{d0 d1 d'0 d'1} = RQ(RQ)† = RQQ†R† = RR† = UΛU† → R = Usqrt(Λ)
        #job_list = [B01B01†, B12B12†, A2A2†, B02B02†]
        subscript_list = ["hbai,hBAi->baBA", "kldc,klDC->dcDC", "dmne,DmnE->deDE", "eopf,EopF->efEF"]
    
        if WORLD_MPI_RANK == 0:
            operand_list = [[xp.conj(B), B], [xp.conj(B), B], [xp.conj(A), A], [xp.conj(B), B]]
        else:
            operand_list = None
        AA, BB, CC, DD = env_tensor_components(subscript_list, operand_list)
    
        if WORLD_MPI_RANK == 0:
            C = AA
            D = oe.contract("acAC,abAB,bdBD->cCdD", BB, CC, DD)
        else:
            C, D = None, None
        del AA, BB, CC, DD, operand_list
    
        subscript_right = "abAB,cCdD,bdei,BDek,fcaj,fCAl->ijkl"
        operands_right = [C, D, A, xp.conj(A), A, xp.conj(A)]
        del B, C, D
        gpu_syn(use_gpu)
        comm.barrier()
        #QR right---------------------------------------------------------------------------


    elif direction == 'Y' or direction == 'y':
        #QR left----------------------------------------------------------------------------
        #Env_{y'01 y'02, y01 y02} = (QR)†QR = R†Q†QR = R†R = UΛU† → R = sqrt(Λ)U†)
        #job_list = [A0†A0, A1†A1, B12†B12, A2†A2]
        subscript_list = ["abgh,ABgh->abAB", "jdck,jDCk->dcDC", "lmed,lmED->edED", "enof,EnoF->efEF"]

        if WORLD_MPI_RANK == 0:
            operand_list = [[xp.conj(A), A], [xp.conj(A), A], [B, B], [xp.conj(A), A]]
        else:
            operand_list = None
        AA, BB, CC, DD = env_tensor_components(subscript_list, operand_list)
        
        if WORLD_MPI_RANK == 0:
            C = AA
            D = oe.contract("abAB,caCA,cdCD->bBdD", BB, CC, DD)
        else:
            C, D = None, None
        del AA, BB, CC, DD, operand_list
        
        subscripts_left = "abAB,cCdD,ecai,aCAk,dfjb,DflB->ijkl"
        operands_left = [C, D, B, B, B, B]
        del B, C, D
        gpu_syn(use_gpu)
        comm.barrier()
        #QR left----------------------------------------------------------------------------
        
        #QR right---------------------------------------------------------------------------
        #Env_{l1 u2, l'1 u'2} = RQ(RQ)† = RQQ†R† = RR† = UΛU† → R = Usqrt(Λ)
        #job_list = [A0A0†, B01B01†, B12B12†, B02B02†]
        subscript_list = ["abgh,ABgh->abAB", "icaj,iCAj->caCA", "lmed,lmED->edED", "fopb,FopB->fbFB"]

        if WORLD_MPI_RANK == 0:
            operand_list = [[xp.conj(A), A], [B, B], [B, B], [B, B]]
        else:
            operand_list = None
        AA, BB, CC, DD = env_tensor_components(subscript_list, operand_list)

        if WORLD_MPI_RANK == 0:
            C = CC
            D = oe.contract("abAB,cbCB,dcDC->aAdD", DD, AA, BB)
        else:
            C, D = None, None
        del AA, BB, CC, DD, operand_list

        subscript_right = "abAB,cCdD,idbe,kBDe,ajfc,AlfC->ijkl"
        operands_right = [C, D, A, xp.conj(A), A, xp.conj(A)]
        del B, C, D
        gpu_syn(use_gpu)
        comm.barrier()
        #QR right---------------------------------------------------------------------------


    #local_M_left = xp.zeros(shape=(Dcut, Dcut, Dcut, Dcut), dtype=complex)
    #for n, 

        
def rsvd_for_3d_SU2_pure_gauge_initial_tensor(A, k:int, comm:MPI.Intercomm, seed=None):
    rank = comm.Get_rank()
    seed = rank + seed if seed is not None else rank
    if type(A) == np.ndarray:
        qr  = np.linalg.qr
        svd = np.linalg.svd
        rs = np.random.RandomState(seed)
    else:
        from cupy.linalg import svd as svd
        from cupy.linalg import qr  as qr
        from cupy import random
        rs = random.RandomState(seed)