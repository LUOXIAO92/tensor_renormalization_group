import time
import math
import copy

try:
    import cupy as cp
except:
    pass
import numpy as np

import opt_einsum as oe
from itertools import product

from mpi4py import MPI

from tools.mpi_tools import contract_slicer, pure_gauge_slice, gpu_syn, flatten_2dim_job_results
from tools.linalg_tools import eigh, svd

class SU2_pure_gauge:
    def __init__(self, 
                 dim:int, 
                 Dcut:int, 
                 Ks:tuple|list,
                 β:float, 
                 ε:float|None, 
                 comm:MPI.Intracomm, 
                 use_gpu=False):
        
        self.dim  = dim
        self.Dcut = Dcut
        self.Ks   = Ks
        self.β = β
        self.ε = ε

        self.comm    = comm
        self.use_gpu = use_gpu

    def plaquette_tensor(self, chi, chunk:tuple, legs_to_hosvd:list):
        """
        >>>         U1†
        >>>     ----------
        >>>     |        |
        >>>  U0†|        |U2
        >>>     |        |
        >>>     ----------
        >>>         U3
        """

        len_legs_to_hosvd = len(legs_to_hosvd)
        assert (len_legs_to_hosvd == 1) or (len_legs_to_hosvd == 2) or (len_legs_to_hosvd == 4), f"number of legs must be 1,2 or 4"

        for leg in legs_to_hosvd:
             assert legs_to_hosvd[leg] == leg, "legs_to_hosvd must be [0], [0,1] or [0,1,2,3]"

        WORLD_MPI_COMM = self.comm
        WORLD_MPI_RANK = WORLD_MPI_COMM.Get_rank()
        WORLD_MPI_SIZE = WORLD_MPI_COMM.Get_size()        

        from tools.trg_tools import SU2_matrix_Gauss_Legendre_quadrature
        
        U, w, J, I = SU2_matrix_Gauss_Legendre_quadrature(self.Ks[0], self.Ks[1], self.Ks[2], 
                                                          self.comm, to_cupy=self.use_gpu)

        N = self.Ks[0] * self.Ks[1] * self.Ks[2]

        assert len(chunk) == 3, "length of `chunk` must be 3"
        
        WORLD_MPI_COMM.barrier()
        iteration = [
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(N, chunk[0], chunk[1], chunk[2])),
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(chunk[0], N, chunk[1], chunk[2])),
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(chunk[0], chunk[1], N, chunk[2])),
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(chunk[0], chunk[1], chunk[2], N)),
                     ]
        WORLD_MPI_COMM.barrier()


        #compute hosvd P_{U0, U1, U2, U3} = s_{a, b, c, d} V_{U0, a} V_{U1, b} V_{U2, c} V_{U3, d}
        #compute M_{U0, U0'} = P_{U0, U1, U2, U3} P*_{U'0, U1, U2, U3}, ..., M_{U3, U3'}
        M_list = []
        for i in range(4):
            if i <= legs_to_hosvd[-1]:
                WORLD_MPI_COMM.barrier()
                gpu_syn(self.use_gpu)
                leg_hosvd = legs_to_hosvd[i]
                M_tmp = plaquette_contraction_for_hosvd(self.β, self.ε, U, w, J, leg_hosvd, iteration[leg_hosvd], 
                                                        self.comm, use_gpu=self.use_gpu, verbose=True)
                M_list.append(M_tmp)
                WORLD_MPI_COMM.barrier()
                if WORLD_MPI_RANK == 0:
                    print(f"Calculation of leg U{i} finished.")
                del M_tmp
            else:
                M_list.append(M_list[i % len_legs_to_hosvd])
                

        WORLD_MPI_COMM.barrier()
        if WORLD_MPI_RANK == 0:
            M =[]
            for rank in range(WORLD_MPI_SIZE):
                M_send = []
                for i in range(rank, 4, WORLD_MPI_SIZE):
                    if rank == 0:
                        M.append(M_list[i])
                    else:
                        M_send.append(M_list[i])
                if rank != 0:
                    WORLD_MPI_COMM.send(obj=M_send, dest=rank, tag=rank)
        else:
            gpu_syn(self.use_gpu)
            M = WORLD_MPI_COMM.recv(source=0, tag=WORLD_MPI_RANK)
        WORLD_MPI_COMM.barrier()


        WORLD_MPI_COMM.barrier()
        gpu_syn(self.use_gpu)
        eigvals, vs = [], []
        for m in M:
            if m is not None:
                e, v = eigh(m, shape=[[0], [1]], k=chi, truncate_eps=1e-10, degeneracy_eps=1e-5)
                eigvals.append(e)
                vs.append(v)
        WORLD_MPI_COMM.barrier()
        

        WORLD_MPI_COMM.barrier()
        for i, e in enumerate(eigvals):
            leg = i*WORLD_MPI_SIZE + WORLD_MPI_RANK
            print(f"eigen values for {leg}th leg at rank{WORLD_MPI_RANK} are", e)
        WORLD_MPI_COMM.barrier()

        #Tensor P_{U0, U1, U2, U3} = s_{a, b, c, d} V_{U0, a} V_{U1, b} V_{U2, c} V_{U3, d}

        all_vs = []
        for rank in range(WORLD_MPI_SIZE):
            if rank == WORLD_MPI_RANK:
                buf = vs
            else:
                buf = None
            buf = WORLD_MPI_COMM.bcast(buf, root=rank)
            all_vs.append(buf)
        del vs
        WORLD_MPI_COMM.barrier()

        vs = flatten_2dim_job_results(all_vs, 4, WORLD_MPI_COMM)
        
        iteration = pure_gauge_slice(shape=(N, N, N, N), chunk=(N, chunk[0], chunk[1], chunk[2]))
        #Q0, Q1, Q2, Q3 = cp.conj(vs[0].T), cp.conj(vs[1].T), cp.conj(vs[2].T), cp.conj(vs[3].T)
        Q0, Q2 = cp.conj(vs[0].T), vs[2].T
        Q1, Q3 = cp.conj(vs[1].T), vs[3].T

        I02 = oe.contract("iU,jU->ij", Q0, Q2)
        I13 = oe.contract("iU,jU->ij", Q1, Q3)
        TrI02 = cp.trace(I02)
        normI02sqr = cp.linalg.norm(I02)**2
        TrI13 = cp.trace(I13)
        normI13sqr = cp.linalg.norm(I13)**2
        print(f"At rank {WORLD_MPI_RANK}. Tr(V_{{U,c0}}V_{{U,a0}})={TrI02}, norm(V_{{U,c0}}V_{{U,a0}})={normI02sqr},Tr(V_{{U,d0}}V_{{U,b0}})={TrI13}, norm(V_{{U,d0}}V_{{U,b0}})={normI13sqr}")

        T_shape = (Q0.shape[0], Q1.shape[0], Q2.shape[0], Q3.shape[0])
        T_local = cp.zeros(shape=T_shape, dtype=complex)
        t0 = time.time() if WORLD_MPI_RANK == 0 else None
        for n, i in enumerate(iteration):
            i0, i1, i2, i3 = i
            if n % WORLD_MPI_SIZE == WORLD_MPI_RANK:
        
                #TrP = oe.contract("abi,bcj,dck,adl->ijkl", cp.conj(U[:,:,i0]), cp.conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
                TrP = oe.contract("dci,adj,abk,bcl->ijkl", cp.conj(U[:,:,i0]), cp.conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
                
                if self.ε is not None:
                    norm, idx = admissibility_condition(TrP, self.ε)
                    P = (1 - 0.5*TrP.real) / (1 - (norm / self.ε))
                    P = cp.exp(-self.β * P)
                    P[idx] = 0.0
                else:
                    P = cp.exp(-self.β * (1 - 0.5*TrP.real))
                
                P = oe.contract("i,j,k,l,i,j,k,l,ijkl->ijkl", cp.sqrt(w[i0]), cp.sqrt(w[i1]), cp.sqrt(w[i2]), cp.sqrt(w[i3]), 
                                                              cp.sqrt(J[i0]), cp.sqrt(J[i1]), cp.sqrt(J[i2]), cp.sqrt(J[i3]), P)
                q0, q1, q2, q3 = Q0[:,i0], Q1[:,i1], Q2[:,i2], Q3[:,i3]
                T_local += oe.contract("ABCD,xA,YB,XC,yD->xXyY", P, q0, q1, q2, q3)
        
                if (n > 0) and (n % (25*WORLD_MPI_SIZE) == 0) and (WORLD_MPI_RANK == 0):
                    t1 = time.time() if WORLD_MPI_RANK == 0 else None
                    print(f"n={n}", end=", ")
                    print(f"{n // WORLD_MPI_SIZE} times finished. Time= {t1-t0:.2e} s. Size of P: {P.nbytes/(1024**3):.2e} Gbytes")
                    t0 = time.time() if WORLD_MPI_RANK == 0 else None
        
        T = WORLD_MPI_COMM.reduce(sendobj=T_local, op=MPI.SUM, root=0)
        gpu_syn(self.use_gpu)
        WORLD_MPI_COMM.barrier()

        return T
    

    def atrg_tensor(self, 
                    chi_plaquette:int, 
                    chi_atrgtensor:int, 
                    chunk_plaquette:tuple, 
                    chunk_environment:tuple,
                    legs_to_hosvd:list, 
                    truncate_eps:float, 
                    degeneracy_eps:float):
        
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        WORLD_MPI_COMM = self.comm

        P = self.plaquette_tensor(chi_plaquette, chunk_plaquette, legs_to_hosvd)
        Q = None
        if WORLD_MPI_COMM.Get_rank() == 0:
            I = xp.diag(xp.ones(shape=chi_plaquette, dtype=P.dtype))
            Q = oe.contract("Ωi,Ωj,Ωk,Ωl->ijkl", I, I, I, I)
            del I
        gpu_syn(self.use_gpu)
        WORLD_MPI_COMM.barrier()

        A, B, C, D = env_tensor_legswapping_3d_gauge(P, Q, 
                                                     chi_plaquette, 
                                                     truncate_eps, 
                                                     degeneracy_eps, 
                                                     self.comm, 
                                                     self.use_gpu)
        env_tensor_3d_gauge_left(chi_plaquette, 
                                 A, B, C, D, P, Q,
                                 "T",
                                 truncate_eps,
                                 degeneracy_eps,
                                 chunk_environment,
                                 self.comm,
                                 self.use_gpu)
        







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
        sqrt = np.sqrt
        abs  = np.abs
    else:
        from cupy import sqrt, abs
    
    norm = 4 - 2 * TrP.real
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
    
    
    gpu_syn(use_gpu)
    WORLD_MPI_COMM.barrier()

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

    M = WORLD_MPI_COMM.reduce(sendobj=M_local, op=MPI.SUM, root=0)
    gpu_syn(use_gpu)
    WORLD_MPI_COMM.barrier()

    return M

def env_tensor_legswapping_3d_gauge(P, Q, chi, truncate_eps:float, degeneracy_eps:float, comm:MPI.Intercomm, use_gpu=False):
    if use_gpu:
        xp = cp
    else:
        xp = np

    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()

    _Qaxes_list = [(3, 0, 1, 2), (0, 1, 2, 3)]
    _Paxes_list = [(0, 1, 2, 3), (1, 2, 3, 0)]
    Njobs = 2
    results = None
    for job_id in range(Njobs):
        dest_rank = job_id % MPI_SIZE

        if MPI_RANK == 0:
            sendjob = [Q, _Qaxes_list[job_id], P, _Paxes_list[job_id]]

            if MPI_RANK != dest_rank:
                comm.send(obj=sendjob, dest=dest_rank, tag=dest_rank)
            else:
                job = sendjob
        else:
            if MPI_RANK == dest_rank:
                job = comm.recv(source=0, tag=MPI_RANK)
            else:
                pass

        if MPI_RANK == dest_rank:
            Q, _Qaxes, P, _Paxes = job
            _Q = xp.transpose(Q, _Qaxes)
            _P = xp.transpose(P, _Paxes)

    for job_id in range(Njobs):
        dest_rank = job_id % MPI_SIZE

        if MPI_RANK == dest_rank:
            if job_id == 0:
                #Q_{u1, x12, t'12, l2} = usQ_{u1, x12, i}, svhQ_{i, t'12, l2}
                #P_{l1, u1, r1, d1} = usP_{l1, u1, j}, svhP_{j, r1, d1}
                us_Q, svh_Q, _ = svd(_Q, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                us_P, svh_P, sP = svd(_P, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                #M_{x12, i, l1, j} = usQ_{x12, u1, i} usP_{l1, u1, j} = M_{i, l1, j, x12}
                M = oe.contract("xui,luj->iljx", us_Q, us_P)
                #M_{i, l1, j, x12} = usM_{i, l1, k} svhM_{k, j, x12}
                usM, svhM, sM = svd(M, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                #A_{t'12, l2, l1, k} = svhQ_{i, t'12, l2} usM_{i, l1, k}
                A = oe.contract("itL,ilk->tLlk", svh_Q, usM)
                #B_{k, d1, x12, r1} = svhM_{k, j, x12} svhP_{j, r1, d1}
                B = oe.contract("kjx,jrd->kdxr", svhM, svh_P)
                results = [A, B]

                #print(f"sP{job_id}:", sP/xp.max(sP))
                #print(f"sM{job_id}:", sM/xp.max(sM))

            elif job_id == 1:
                #Q_{y20, u0, d2, t'20} = usQ_{y20, u0, i}, svhQ_{i, d2, t'20}
                #P_{u0, r0, d0, l0} = usP_{u0, r0, j}, svhP_{j, d0, l0}
                us_Q, svh_Q, _ = svd(_Q, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                us_P, svh_P, sP = svd(_P, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                #M_{i, r0, j, y20} = usQ_{y20, u0, i} usP_{u0, r0, j} = M_{i, r0, j, y20}
                M = oe.contract("yui,urj->irjy", us_Q, us_P)
                #M_{i, r0, j, y20} = usM_{i, r0, k} svhM_{k, j, y20}
                usM, svhM, sM = svd(M, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                #C_{k, l0, d0, y20} = svh_P_{d0, l0, j} svhM_{k, y20, j}
                C = oe.contract("kjy,jdl->kdly", svhM, svh_P)
                #D_{t'20, d2, r0, k} = svhQ_{i, d2, t'20} usM_{i, r0, k}
                D = oe.contract("idt,irk->tdrk", svh_Q, usM)
                results = [C, D]

                #print(f"sP{job_id}:", sP/xp.max(sP))
                #print(f"sM{job_id}:", sM/xp.max(sM))

    results = comm.gather(sendobj=results, root=0)
    if MPI_RANK == 0:
        results_1dim = []
        for job_id in range(Njobs):
            for result in results[job_id]:
                results_1dim.append(result)
    else:
        results_1dim = [None, None, None, None]
    del results

    return results_1dim


def env_tensor_3d_gauge_left(chi, A, B, C, D, P, Q, 
                             direction:str, 
                             truncate_eps:float, 
                             degeneracy_eps:float,
                             Env_chunks:tuple,
                             comm:MPI.Intercomm, 
                             use_gpu=False):
    if use_gpu:
        xp = cp
    else:
        xp = np

    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()

    if MPI_RANK == 0:
        #Part(1)_{aAbB} = B_{a, d1, x12, r1}  Q_{x01, r1, l0, y01} C_{b, l0, y20, d0}
        #                B†_{A, d1, x12, R1} Q_{x01, R1, L0, y01} C†_{B, L0, y20, d0}
        part1 = oe.contract("icda,jcdA,eabf,eABf,kgbh,lgBh->ijkl", B, xp.conj(B), Q, Q, C, xp.conj(C))
    else:
        part1 = None

    if direction == "T" or direction == "t":
        if MPI_RANK == 0:
            #Part(2)_{l2, L2, d2, D2} = P_{l2, u2, r2, d2} P†_{L2, u2, r2, D2} 
            part2 = oe.contract("lurd,LurD->lLdD", P, xp.conj(P))
        else:
            part2 = None

        Njobs = 2
        results = []
        parts = [part1, part2]
        for job_id in range(Njobs):
            dest_rank = job_id % MPI_SIZE

            if MPI_RANK == 0:
                sendjob = parts[job_id]

                if MPI_RANK == dest_rank:
                    job = sendjob
                else:
                    comm.send(sendjob, dest=dest_rank, tag=dest_rank)
            
            else:
                if MPI_RANK == dest_rank:
                    job = comm.recv(source=0, tag=MPI_RANK)
        
        for job_id in range(Njobs):
            dest_rank = job_id % MPI_SIZE
            if MPI_RANK == dest_rank:
                us, svh, s = svd(job, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                us  = xp.reshape(us , (chi, chi, us.shape[2]))
                svh = xp.reshape(svh, (svh.shape[0], chi, chi))
                results.append([us, svh])
                #print(f"singular values of part{job_id} are ", s/xp.max(s))
        gpu_syn(use_gpu)
        comm.barrier()

        results = comm.gather(results, root=0)
        if MPI_RANK == 0:
            Envs = flatten_2dim_job_results(results, job_size=Njobs, comm=comm)
        else:
            Envs = [None, None]
        del results, part1, part2
        gpu_syn(use_gpu)
        comm.barrier()

        slicing = contract_slicer(shape=(chi, chi, chi, chi, chi), 
                                  chunk=Env_chunks, 
                                  comm=comm)
        ops = None
        if MPI_RANK == 0:
            us1, svh1 = Envs[0]
            us2, svh2 = Envs[1]
            ops  = [A, us1, us2, D, svh1, svh2]
        ops = comm.bcast(ops, root=0)
        A, us1, us2, D, svh1, svh2 = ops
        _, _, chii = us1.shape
        _, _, chij = us2.shape
        subs = ["tlba,TlBA,aAi,bBj->tTij", "tbra,TBrA,iaA,jbB->tTij"]

        for n, (a, aa, b, bb, l) in enumerate(slicing):
            if n == 0:
                path, _ = oe.contract_path(subs[0], A[:,l,b,a], xp.conj(A)[:,l,bb,aa], us1[a,aa,:], us2[b,bb,:])
                break
        
        Env1 = xp.zeros(shape=(chi, chi, chii, chij), dtype=A.dtype)
        t0 = time.time()
        for n, (a, aa, b, bb, l) in enumerate(slicing):
            if n % MPI_SIZE == MPI_RANK:
                Env1 += oe.contract(subs[0], 
                                    A[:,l,b,a], xp.conj(A)[:,l,bb,aa], 
                                    us1[a,aa,:], us2[b,bb,:], optimize=path)
            if n % 4 == 4 - 1 and MPI_RANK == 0:
                t1 = time.time()
                print(f"Global iters {n+1}, local iters {n//MPI_SIZE+1}, time= {t1-t0:.2e} s")
                t0 = time.time()
        Env1 = comm.reduce(Env1, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()


        slicing = contract_slicer(shape=(chi, chi, chi, chi, chi), 
                                  chunk=Env_chunks, 
                                  comm=comm)
        for n, (a, aa, b, bb, r) in enumerate(slicing):
            if n == 0:
                path, _ = oe.contract_path(subs[1], D[:,b,r,a], xp.conj(D)[:,bb,r,aa], svh1[:,a,aa], svh2[:,b,bb])
                break
        Env2 = xp.zeros(shape=(chi, chi, chii, chij), dtype=A.dtype)
        t0 = time.time()
        for n, (a, aa, b, bb, r) in enumerate(slicing):
            if n % MPI_SIZE == MPI_RANK:
                Env2 += oe.contract(subs[1], 
                                    D[:,b,r,a], xp.conj(D)[:,bb,r,aa], 
                                    svh1[:,a,aa], svh2[:,b,bb], optimize=path)
            if n % 4 == 4 - 1 and MPI_RANK == 0:
                t1 = time.time()
                print(f"Global iters {n+1}, local iters {n//MPI_SIZE+1}, time= {t1-t0:.2e} s")
                t0 = time.time()
        Env2 = comm.reduce(Env2, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()

        if MPI_RANK == 0:
            Env = oe.contract("ikab,jlab->ijkl", Env1, Env2)
            mat = xp.reshape(Env, (chi*chi, chi*chi))
            hermiterr = xp.linalg.norm(mat - xp.conj(mat.T))
            print(f"hermit err={hermiterr}")
        else:
            Env = None

    if direction == "X" or direction == "x":
        if MPI_RANK == 0:
            #Part(2)_{l2,L2,a,A} = A_{t'12, l1, l2, a} A†_{t'12, l1, L2, A}
            part2 = oe.contract("tila,tiLA->lLaA", A, xp.conj(A))
            Env12 = oe.contract("lLaA,aAbB->lLbB", part2, part1)
            us, svh, _ = svd(Env12, shape=[[0,1], [2,3]])
        else:
            us, svh = None, None

        subs = []
        parts = [us, svh]
        results = []
        Njobs = 2
        for job_id in range(Njobs):
            dest_rank = job_id % MPI_SIZE

            if MPI_RANK == 0:
                sendjob = parts[job_id]

                if MPI_RANK == dest_rank:
                    job = sendjob
                else:
                    comm.send(sendjob, dest=dest_rank, tag=dest_rank)
            
            else:
                job = comm.recv(source=0, tag=MPI_RANK)
        
        for job_id in range(Njobs):
            dest_rank = job_id % MPI_SIZE
            if MPI_RANK ==  dest_rank:
                results.append(
                    #Env(1)_{}
                    oe.contract("")
                )

        



    import sys
    sys.exit(0)


def env_tensor_3d_SU2_pure_gauge(A, direction:str, chunk:tuple, comm:MPI.Intercomm, use_gpu=False, low_communication_cost=False, verbose=False):
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
            
    if WORLD_MPI_RANK == 0:
        chi = A.shape[0]
        I = xp.diag([1.0 for _ in range(chi)])
        B = oe.contract("Ωi,Ωj,Ωk,Ωl->ijkl", I, I, I, I)
        B = B.astype(complex)
    else:
        chi = None
        I, B = None, None
    del I
    chi = comm.bcast(obj=chi, root=0)


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
                    comm.send(obj=sendjob, dest=dest_rank, tag=dest_rank)
                else:
                    job = sendjob
                    
            #Recive job from rank 0
            else:
                if WORLD_MPI_RANK == dest_rank:
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
            Envs = [None for _ in range(4)]
        del results
        gpu_syn(use_gpu)
        comm.barrier()

        return Envs

    shape = (chi, chi, chi, chi)
    slices = [[slice(j, min(shape[i], j+chunk[i])) for j in range(0, shape[i], chunk[i])] for i in range(4)]
    noslice = [slice(0, chi)]
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
        
        legC = product(noslice, noslice, slices[0], slices[1])
        legD = product(noslice, slices[2], noslice, slices[3])
        leg0 = product(noslice, noslice, noslice, noslice)
        leg1 = product(slices[1], noslice, noslice, slices[2])
        leg2 = product(noslice, noslice, noslice, noslice)
        leg3 = product(noslice, noslice, slices[0], slices[3])
        iter_left = product(legC, legD, leg0, leg1, leg2, leg3)
        subscripts_left = "abAB,cCdD,biec,BkeC,fjad,flAD->ijkl"
        if WORLD_MPI_RANK == 0:
            operands_left = [C, D, B, B, B, B]
        else:
            operands_left = None
        del C, D, legC, legD, leg0, leg1, leg2, leg3
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

        legC = product(noslice, noslice, slices[0], slices[1])
        legD = product(noslice, slices[2], noslice, slices[3])
        leg0 = product(noslice, noslice, noslice, noslice)
        leg1 = product(slices[1], slices[3], noslice, noslice)
        leg2 = product(noslice, noslice, noslice, noslice)
        leg3 = product(noslice, slices[2], slices[0], noslice)
        iter_right = product(legC, legD, leg0, leg1, leg2, leg3)
        subscript_right = "abAB,cCdD,bdei,BDek,fcaj,fCAl->ijkl"
        if WORLD_MPI_RANK == 0:
            operands_right = [C, D, A, xp.conj(A), A, xp.conj(A)]
        else:
            operands_right = None
        del B, C, D, legC, legD, leg0, leg1, leg2, leg3
        gpu_syn(use_gpu)
        comm.barrier()

        #QR right---------------------------------------------------------------------------


    elif direction == 'X' or direction == 'x':
        #QR left----------------------------------------------------------------------------
        #Env_{r'0 r'2, r0 r2} = (QR)†QR = R†Q†QR = R†R = UΛU† → R = sqrt(Λ)U†)
        #job_list = [B01B01†, A1A1†, B12B12†, B02B02†]
        subscript_list = ["hcai,hCAi->caCA", "idck,iDCk->dcDC", "lmed,lmED->edED", "fopb,FopB->fbFB"]
    
        if WORLD_MPI_RANK == 0:
            operand_list = [[B, B], [xp.conj(A), A], [B, B], [B, B]]
        else:
            operand_list = None
        AA, BB, CC, DD = env_tensor_components(subscript_list, operand_list)
        
        if WORLD_MPI_RANK == 0:
            C = DD
            D = oe.contract("abAB,caCA,dcDC->bBdD", AA, BB, CC)
        else:
            C, D = None, None
        del AA, BB, CC, DD, operand_list
        
        legC = product(noslice, noslice, slices[0], slices[1])
        legD = product(noslice, slices[2], noslice, slices[3])
        leg0 = product(noslice, noslice, noslice, noslice)
        leg1 = product(slices[2], slices[1], noslice, noslice)
        leg2 = product(noslice, noslice, noslice, noslice)
        leg3 = product(slices[3], noslice, noslice, slices[0])
        iter_left = product(legC, legD, leg0, leg1, leg2, leg3)
        subscripts_left = "abAB,cCdD,cbie,CBke,dfja,DflA->ijkl"
        if WORLD_MPI_RANK == 0:
            operands_left = [C, D, xp.conj(A), A, xp.conj(A), A]
        else:
            operands_left = None
        del C, D, legC, legD, leg0, leg1, leg2, leg3
        gpu_syn(use_gpu)
        comm.barrier()
        #QR left----------------------------------------------------------------------------
        
        #QR right---------------------------------------------------------------------------
        #Env_{d0 d1 d'0 d'1} = RQ(RQ)† = RQQ†R† = RR† = UΛU† → R = Usqrt(Λ)
        #job_list = [A0A0†, A1A1†, A2A2†, B02B02†]
        subscript_list = ["abgh,ABgh->abAB", "jdck,jDCk->dcDC", "emnf,EmnF->efEF", "fopb,FopB->fbFB"]
    
        if WORLD_MPI_RANK == 0:
            operand_list = [[xp.conj(A), A], [xp.conj(A), A], [xp.conj(A), A], [B, B]]
        else:
            operand_list = None
        AA, BB, CC, DD = env_tensor_components(subscript_list, operand_list)
    
        if WORLD_MPI_RANK == 0:
            C = BB
            D = oe.contract("abAB,acAC,dcDC->aAdD", CC, DD, AA)
        else:
            C, D = None, None
        del AA, BB, CC, DD, operand_list
    
        legC = product(noslice, noslice, slices[0], slices[1])
        legD = product(noslice, slices[2], noslice, slices[3])
        leg0 = product(noslice, noslice, noslice, noslice)
        leg1 = product(noslice, slices[1], slices[3], noslice)
        leg2 = product(noslice, noslice, noslice, noslice)
        leg3 = product(noslice, noslice, slices[2], slices[0])
        iter_right = product(legC, legD, leg0, leg1, leg2, leg3)
        subscript_right = "abAB,cCdD,ibde,kBDe,jfca,lfCA->ijkl"
        if WORLD_MPI_RANK == 0:
            operands_right = [C, D, B, B, B, B]
        else:
            operands_right = None
        del B, C, D, legC, legD, leg0, leg1, leg2, leg3
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
        
        legC = product(noslice, noslice, slices[0], slices[1])
        legD = product(noslice, slices[2], noslice, slices[3])
        leg0 = product(noslice, noslice, noslice, noslice)
        leg1 = product(noslice, slices[2], slices[0], noslice)
        leg2 = product(noslice, noslice, noslice, noslice)
        leg3 = product(slices[3], noslice, noslice, slices[1])
        iter_left = product(legC, legD, leg0, leg1, leg2, leg3)
        subscripts_left = "abAB,cCdD,ecai,aCAk,dfjb,DflB->ijkl"
        if WORLD_MPI_RANK == 0:
            operands_left = [C, D, B, B, B, B]
        else:
            operands_left = None
        del C, D, legC, legD, leg0, leg1, leg2, leg3
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

        legC = product(noslice, noslice, slices[0], slices[1])
        legD = product(noslice, slices[2], noslice, slices[3])
        leg0 = product(noslice, noslice, noslice, noslice)
        leg1 = product(noslice, slices[1], slices[3], noslice)
        leg2 = product(noslice, noslice, noslice, noslice)
        leg3 = product(slices[0], noslice, noslice, slices[2])
        iter_right = product(legC, legD, leg0, leg1, leg2, leg3)
        subscript_right = "abAB,cCdD,idbe,kBDe,ajfc,AlfC->ijkl"
        if WORLD_MPI_RANK == 0:
            operands_right = [C, D, A, xp.conj(A), A, xp.conj(A)]
        else:
            operands_right = None
        del B, C, D, legC, legD, leg0, leg1, leg2, leg3
        gpu_syn(use_gpu)
        comm.barrier()
        #QR right---------------------------------------------------------------------------


    if WORLD_MPI_RANK == 0:
        iter_left_  = copy.copy(iter_left)
        for n, (legc, legd, leg0, leg1, leg2, leg3) in enumerate(iter_left_):
            if n == 0:
                path_left, _ = oe.contract_path(subscripts_left, 
                                                operands_left[0][legc], 
                                                operands_left[1][legd],
                                                operands_left[2][leg0],
                                                operands_left[3][leg1],
                                                operands_left[4][leg2],
                                                operands_left[5][leg3],
                                                optimize='optimal')
                break

        iter_right_ = copy.copy(iter_right)
        for n, (legc, legd, leg0, leg1, leg2, leg3) in enumerate(iter_right_):
            if n == 0:
                path_right, _ = oe.contract_path(subscript_right, 
                                                 operands_right[0][legc], 
                                                 operands_right[1][legd],
                                                 operands_right[2][leg0],
                                                 operands_right[3][leg1],
                                                 operands_right[4][leg2],
                                                 operands_right[5][leg3],
                                                 optimize='optimal')
                break
        
        del iter_left_, iter_right_
    else:
        path_left  = None
        path_right = None
    
    path_left  = comm.bcast(path_left , root=0)
    path_right = comm.bcast(path_right, root=0)

    if low_communication_cost:
        #Calculate Rleft----------------------------------------------------------
        operands_left   = comm.bcast(operands_left  , root=0)
        subscripts_left = comm.bcast(subscripts_left, root=0)
        iter_left       = comm.bcast(iter_left      , root=0)
        E_left_local = xp.zeros(shape=shape, dtype=complex)
        gpu_syn(use_gpu)
        comm.barrier()
        t0  = time.time()
        t00 = time.time()
        for n, (legc, legd, leg0, leg1, leg2, leg3) in enumerate(iter_left):
            if n % WORLD_MPI_SIZE == WORLD_MPI_RANK:
                E_left_local += oe.contract(subscripts_left, 
                                            operands_left[0][legc], 
                                            operands_left[1][legd],
                                            operands_left[2][leg0],
                                            operands_left[3][leg1],
                                            operands_left[4][leg2],
                                            operands_left[5][leg3],
                                            optimize=path_left)
                
            if verbose:
                if n % chi == (chi - 1) and WORLD_MPI_RANK == 0:
                    t1 = time.time()
                    print(f"Local iterations {n+1}, time= {t1-t0:.2e} s.")
                    t0 = time.time()
            
        E_left = comm.reduce(sendobj=E_left_local, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()
        t11 = time.time()
        del E_left_local, operands_left, subscripts_left, iter_left

        print(f"Total iterations {n}, time= {t11-t00:.2e} s.")
        #Calculate Rleft----------------------------------------------------------


        #Calculate Rright----------------------------------------------------------
        operands_right   = comm.bcast(operands_right  , root=0)
        subscripts_right = comm.bcast(subscripts_right, root=0)
        iter_right       = comm.bcast(iter_right      , root=0)
        E_right_local = xp.zeros(shape=shape, dtype=complex)
        gpu_syn(use_gpu)
        comm.barrier()
        t0  = time.time()
        t00 = time.time()
        for n, (legc, legd, leg0, leg1, leg2, leg3) in enumerate(iter_right):
            if n % WORLD_MPI_SIZE == WORLD_MPI_RANK:
                E_right_local += oe.contract(subscripts_right, 
                                             operands_right[0][legc], 
                                             operands_right[1][legd],
                                             operands_right[2][leg0],
                                             operands_right[3][leg1],
                                             operands_right[4][leg2],
                                             operands_right[5][leg3],
                                             optimize=path_right)
                
            if verbose:
                if n % chi*chi == (chi*chi - 1) and WORLD_MPI_RANK == 0:
                    t1 = time.time()
                    print(f"Local iterations {n}, time= {t1-t0:.2e} s.")
                    t0 = time.time()
            
        E_right = comm.reduce(sendobj=E_right_local, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()
        t11 = time.time()
        del E_right_local, operands_right, subscripts_right, iter_right

        print(f"Total iterations {n}, time= {t11-t00:.2e} s.")
        #Calculate Rright----------------------------------------------------------


    else:
        #Calculate Rleft----------------------------------------------------------
        subscripts_left = comm.bcast(subscripts_left, root=0)
        iter_left       = comm.bcast(iter_left      , root=0)
        E_left_local = xp.zeros(shape=shape, dtype=complex)
        gpu_syn(use_gpu)
        comm.barrier()
        t0  = time.time()
        t00 = time.time()
        for n, (legc, legd, leg0, leg1, leg2, leg3) in enumerate(iter_left):

            dest_rank = n % WORLD_MPI_SIZE
            if WORLD_MPI_RANK == 0:
                sendoperands = [operands_left[0][legc], 
                                operands_left[1][legd],
                                operands_left[2][leg0],
                                operands_left[3][leg1],
                                operands_left[4][leg2],
                                operands_left[5][leg3]]
                if WORLD_MPI_RANK == dest_rank:
                    ops_left = sendoperands
                else:
                    ops_left = comm.send(obj=sendoperands, dest=dest_rank, tag=dest_rank)
            else:
                ops_left = comm.recv(source=0, tag=WORLD_MPI_RANK)

            if WORLD_MPI_RANK == dest_rank:
                E_left_local += oe.contract(subscripts_left, 
                                            ops_left[0], 
                                            ops_left[1],
                                            ops_left[2],
                                            ops_left[3],
                                            ops_left[4],
                                            ops_left[5],
                                            optimize=path_left)
                
            if verbose:
                if n % chi*chi == (chi*chi - 1) and WORLD_MPI_RANK == 0:
                    t1 = time.time()
                    print(f"Local iterations {n}, time= {t1-t0:.2e} s.")
                    t0 = time.time()
            
        E_left = comm.reduce(sendobj=E_left_local, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()
        t11 = time.time()
        del E_left_local, operands_left, subscripts_left, iter_left

        print(f"Total iterations {n}, time= {t11-t00:.2e} s.")
        #Calculate Rleft----------------------------------------------------------


        #Calculate Rright----------------------------------------------------------
        subscripts_right = comm.bcast(subscripts_right, root=0)
        iter_right       = comm.bcast(iter_right      , root=0)
        E_right_local = xp.zeros(shape=shape, dtype=complex)
        gpu_syn(use_gpu)
        comm.barrier()
        t0  = time.time()
        t00 = time.time()
        for n, (legc, legd, leg0, leg1, leg2, leg3) in enumerate(iter_right):

            dest_rank = n % WORLD_MPI_SIZE
            if WORLD_MPI_RANK == 0:
                sendoperands = [operands_right[0][legc], 
                                operands_right[1][legd],
                                operands_right[2][leg0],
                                operands_right[3][leg1],
                                operands_right[4][leg2],
                                operands_right[5][leg3]]
                if WORLD_MPI_RANK == dest_rank:
                    ops_right = sendoperands
                else:
                    ops_right = comm.send(obj=sendoperands, dest=dest_rank, tag=dest_rank)
            else:
                ops_right = comm.recv(source=0, tag=WORLD_MPI_RANK)

            if WORLD_MPI_RANK == dest_rank:
                E_right_local += oe.contract(subscripts_right, 
                                            ops_right[0], 
                                            ops_right[1],
                                            ops_right[2],
                                            ops_right[3],
                                            ops_right[4],
                                            ops_right[5],
                                            optimize=path_right)
                
            if verbose:
                if n % chi*chi == (chi*chi - 1) and WORLD_MPI_RANK == 0:
                    t1 = time.time()
                    print(f"Local iterations {n}, time= {t1-t0:.2e} s.")
                    t0 = time.time()
            
        E_right = comm.reduce(sendobj=E_right_local, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()
        t11 = time.time()
        del E_right_local, operands_right, subscripts_right, iter_right

        print(f"Total iterations {n}, time= {t11-t00:.2e} s.")
        #Calculate Rright----------------------------------------------------------

    

    return E_left, E_right

def squeezer_3d_SU2_pure_gauge_initial_tensor(Dcut, Eleft, Eright, 
                                                  truncate_eps:float, degeneracy_eps:float, 
                                                  comm:MPI.Intercomm, use_gpu=False):
    if use_gpu:
        import cupy as xp
    else:
        xp = np

    MPI_SIZE = comm.Get_size()
    MPI_RANK = comm.Get_rank()

    Njobs = 2
    if MPI_RANK == 0:
        chi_l = Eleft.shape
        chi_r = Eright.shape
        Eleft  = xp.reshape(Eleft , (chi_l[0]*chi_l[1], chi_l[2]*chi_l[3]))
        Eright = xp.reshape(Eright, (chi_r[0]*chi_r[1], chi_r[2]*chi_r[3]))
        Envs = [Eleft, Eright]

        Envs_local = []
        for rank in range(MPI_SIZE):
            Envs_send = []
            for job_id in range(rank, Njobs, MPI_SIZE):
                if rank == 0:
                    Envs_local.append(Envs[job_id])
                else:
                    Envs_send.append(Envs[job_id])
            if rank != 0:
                comm.send(Envs_send)
    else:
        chi_l, chi_r = None, None
        Envs_local = comm.recv(source=0, tag=MPI_RANK)

    chi_l = comm.bcast(chi_l, root=0)
    chi_r = comm.bcast(chi_r, root=0)
    gpu_syn(use_gpu)
    comm.barrier()

    es, us = [], []
    for Env in Envs_local:
        if Env is not None:
            e, u = eigh(Env, shape=[[Env.shape[0]], [Env.shape[1]]], k=min(*Env.shape), 
                        truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps)
            es.append(e)
            us.append(u)
    gpu_syn(use_gpu)
    comm.barrier()

    es = comm.gather(sendobj=es, root=0)
    us = comm.gather(sendobj=us, root=0)
    es = flatten_2dim_job_results(es, 2, comm)
    us = flatten_2dim_job_results(us, 2, comm)
    gpu_syn(use_gpu)
    comm.barrier()

    if MPI_RANK == 0:
        eigvall , eigvalr  = es
        eigvectl, eigvectr = us
        Rleft  = oe.contract("ia,a->ai", xp.conj(eigvectl), xp.sqrt(eigvall))
        Rright = oe.contract("ia,a->ia", eigvectr, xp.sqrt(eigvalr))

        RlRr = Rleft @ Rright
        k = min(*RlRr, Dcut)
        U, S, VH = svd(RlRr, shape=[[RlRr.shape[0]], [RlRr.shape[1]]], k=k, 
                       truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps)
        UH = xp.conj(U.T)
        Sinv = 1 / S
        V = xp.conj(VH.T)

        print("eL",eigvall[:k])
        print("eR",eigvalr[:k])
        print("S", S[:k])

        del U, S, VH
        Pl = oe.contract("ia,aj,j->ij", Rright, V , xp.sqrt(Sinv))
        Pr = oe.contract("i,ia,aj->ij", xp.sqrt(Sinv), UH, Rleft)

        shape_Pl = k, chi_l[2], chi_l[3]
        shape_Pr = chi_r[0], chi_r[1], k
        Pl = xp.reshape(Pl, shape_Pl)
        Pr = xp.reshape(Pr, shape_Pr)
    else:
        Pl, Pr = None, None

    return Pl, Pr
        
def rsvd_3d_SU2_pure_gauge_initial_tensor(A, k:int, comm:MPI.Intercomm, seed=None):
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