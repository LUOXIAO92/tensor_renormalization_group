import time
import cupy as cp
#import numpy as cp
import opt_einsum as oe
import math

from mpi4py import MPI

from tools.mpi_tools import contract_slice, pure_gauge_slice, gpu_syn, flatten_2dim_job_results
from tools.linalg_tools import eigh, svd

class SU2_pure_gauge:
    def __init__(self, 
                 dim:int, 
                 Dcut:int, 
                 Ks:tuple|list,
                 beta:float, 
                 epsilon:float|None, 
                 comm:MPI.Intracomm, 
                 use_gpu=False):
        
        self.dim  = dim
        self.Dcut = Dcut
        self.Ks   = Ks
        self.beta = beta
        self.epsilon = epsilon

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

        from tools.trg_tools import (SU2_matrix_Gauss_Legendre_quadrature, 
                                     admissibility_condition,
                                     plaquette_contraction_for_hosvd)
        
        U, w, J, I = SU2_matrix_Gauss_Legendre_quadrature(self.Ks[0], self.Ks[1], self.Ks[2], 
                                                          self.comm, to_cupy=self.use_gpu)

        N = self.Ks[0] * self.Ks[1] * self.Ks[2]

        assert len(chunk) == 3, "length of `chunk` must be 3"
        
        WORLD_MPI_COMM.Barrier()
        iteration = [
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(N, chunk[0], chunk[1], chunk[2])),
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(chunk[0], N, chunk[1], chunk[2])),
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(chunk[0], chunk[1], N, chunk[2])),
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(chunk[0], chunk[1], chunk[2], N)),
                     ]
        WORLD_MPI_COMM.Barrier()


        #compute hosvd A_{U0, U1, U2, U3} = s_{a, b, c, d} V_{U0, a} V_{U1, b} V_{U2, c} V_{U3, d}
        #compute M_{U0, U0'} = A_{U0, U1, U2, U3} A*_{U'0, U1, U2, U3}, ..., M_{U3, U3'}
        M_list = []
        for i in range(4):
            if i <= legs_to_hosvd[-1]:
                WORLD_MPI_COMM.Barrier()
                gpu_syn(self.use_gpu)
                leg_hosvd = legs_to_hosvd[i]
                M_tmp = plaquette_contraction_for_hosvd(self.beta, self.epsilon, U, w, J, leg_hosvd, iteration[leg_hosvd], 
                                                        self.comm, use_gpu=self.use_gpu, verbose=True)
                M_list.append(M_tmp)
                WORLD_MPI_COMM.Barrier()
                if WORLD_MPI_RANK == 0:
                    print(f"Calculation of leg U{i} finished.")
                del M_tmp
            else:
                M_list.append(M_list[i % len_legs_to_hosvd])
                

        WORLD_MPI_COMM.Barrier()
        if WORLD_MPI_RANK == 0:
            M =[]
            for rank in range(WORLD_MPI_SIZE):
                M_send = []
                for i in range(rank, 4, WORLD_MPI_SIZE):
                    if rank == 0:
                        print(f"keep {i} on rank{rank}")
                        M.append(M_list[i])
                    else:
                        print(f"send {i} to rank{rank}")
                        M_send.append(M_list[i])
                if rank != 0:
                    WORLD_MPI_COMM.send(obj=M_send, dest=rank, tag=rank)
        else:
            gpu_syn(self.use_gpu)
            M = WORLD_MPI_COMM.recv(source=0, tag=WORLD_MPI_RANK)
        WORLD_MPI_COMM.Barrier()


        WORLD_MPI_COMM.Barrier()
        gpu_syn(self.use_gpu)
        eigvals, vs = [], []
        for m in M:
            e, v = eigh(m, shape=[[0], [1]], k=chi, truncate_eps=1e-10, degeneracy_eps=1e-5)
            #v, e, _ = cp.linalg.svd(m)
            #e = e[::-1]
            #v = v[:,::-1]
            #e = e[:chi]
            #e[e<1e-16] = 1e-16
            #v = v[:,:chi]
            eigvals.append(e)
            vs.append(v)
        WORLD_MPI_COMM.Barrier()
        

        WORLD_MPI_COMM.Barrier()
        for i, e in enumerate(eigvals):
            leg = i*WORLD_MPI_SIZE + WORLD_MPI_RANK
            print(f"eigen values for {leg}th leg at rank{WORLD_MPI_RANK} are", e)
        WORLD_MPI_COMM.Barrier()

        #Tensor A_{U0, U1, U2, U3} = s_{a, b, c, d} V_{U0, a} V_{U1, b} V_{U2, c} V_{U3, d}

        all_vs = []
        for rank in range(WORLD_MPI_SIZE):
            if rank == WORLD_MPI_RANK:
                print(f"buf size{len(vs)}, at rank{WORLD_MPI_RANK}")
                buf = vs
            else:
                buf = None
            buf = WORLD_MPI_COMM.bcast(buf, root=rank)
            all_vs.append(buf)
        del vs
        WORLD_MPI_COMM.barrier()

        print("allvs.size",len(all_vs[0]))
        vs = flatten_2dim_job_results(all_vs, 4, WORLD_MPI_COMM)
        #all_vs_1dim = []
        #for i in range(math.ceil(4 / WORLD_MPI_SIZE)):
        #    for j in range(WORLD_MPI_SIZE):
        #        if i * WORLD_MPI_SIZE + j < 4:
        #            all_vs_1dim.append(all_vs[j][i])
        #        else:
        #            break
        #vs = all_vs_1dim
        #del all_vs_1dim, all_vs

        #WORLD_MPI_COMM.barrier()
        #vs_gather_to_rank0 = WORLD_MPI_COMM.gather(vs, root=0)
        #if WORLD_MPI_RANK == 0:
        #    #for leg in range(4):
        #    #    for ranki in range(WORLD_MPI_SIZE):
        #    #        for rankj in range(ranki, WORLD_MPI_SIZE):
        #    #            #print(f"comparing vs[{ranki}][{leg}] and vs[{rankj}][{leg}]")
        #    #            assert cp.allclose(vs_gather_to_rank0[ranki][leg],
        #    #                               vs_gather_to_rank0[rankj][leg])
        #    for ranki in range(WORLD_MPI_SIZE):
        #        for rankj in range(ranki, WORLD_MPI_SIZE):
        #            #print(f"comparing vs[{ranki}][{0}] and vs[{rankj}][{2}]")
        #            #assert cp.allclose(vs_gather_to_rank0[ranki][0],
        #            #                   vs_gather_to_rank0[rankj][2])
        #            #print(f"comparing vs[{ranki}][{1}] and vs[{rankj}][{3}]")
        #            #assert cp.allclose(vs_gather_to_rank0[ranki][1],
        #            #                   vs_gather_to_rank0[rankj][3])
        #            
        #            print(f"comparing vs[{ranki}][{0}] and vs[{rankj}][{2}]")
        #            print(cp.linalg.norm(vs_gather_to_rank0[ranki][0]-vs_gather_to_rank0[rankj][2]))
        #            print(f"comparing vs[{ranki}][{1}] and vs[{rankj}][{3}]")
        #            print(cp.linalg.norm(vs_gather_to_rank0[ranki][1]-vs_gather_to_rank0[rankj][3]))
        #
        #I02 = oe.contract("Ui,Uj->ij",vs[2], vs[0])
        #I13 = oe.contract("Ui,Uj->ij",vs[3], vs[1])
        #TrI02 = cp.trace(I02)
        #normI02sqr = cp.linalg.norm(I02)**2
        #TrI13 = cp.trace(I13)
        #normI13sqr = cp.linalg.norm(I13)**2
        #print(f"At rank {WORLD_MPI_RANK}. Tr(V_{{U,c0}}V_{{U,a0}})={TrI02}, norm(V_{{U,c0}}V_{{U,a0}})={normI02sqr},Tr(V_{{U,d0}}V_{{U,b0}})={TrI13}, norm(V_{{U,d0}}V_{{U,b0}})={normI13sqr}")


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
        
                TrP = oe.contract("abi,bcj,dck,adl->ijkl", cp.conj(U[:,:,i0]), cp.conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
                
                if self.epsilon is not None:
                    #P = oe.contract("abi,bcj,dck,edl->aeijkl", cp.conj(U[:,:,i0]), cp.conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
                    norm, idx = admissibility_condition(TrP, self.epsilon)
                    A = (1 - 0.5*TrP.real) / (1 - norm / self.epsilon)
                    A = cp.exp(-self.beta * A)
                    A[idx] = 0.0
                else:
                    A = cp.exp(-self.beta * (1 - 0.5*TrP))
                
                A = oe.contract("i,j,k,l,i,j,k,l,ijkl->ijkl", cp.sqrt(w[i0]), cp.sqrt(w[i1]), cp.sqrt(w[i2]), cp.sqrt(w[i3]), 
                                                              cp.sqrt(J[i0]), cp.sqrt(J[i1]), cp.sqrt(J[i2]), cp.sqrt(J[i3]), A)
                #A = oe.contract("k,l,k,l,ijkl->ijkl", w[i2], w[i3], J[i2], J[i3], A)
                q0, q1, q2, q3 = Q0[:,i0], Q1[:,i1], Q2[:,i2], Q3[:,i3]
                T_local += oe.contract("ABCD,xA,YB,XC,yD->xXyY", A, q0, q1, q2, q3)
        
                if (n > 0) and (n % (25*WORLD_MPI_SIZE) == 0) and (WORLD_MPI_RANK == 0):
                    t1 = time.time() if WORLD_MPI_RANK == 0 else None
                    print(f"n={n}", end=", ")
                    print(f"{n // WORLD_MPI_SIZE} times finished. Time= {t1-t0:.2e} s. Size of A: {A.nbytes/(1024**3):.2e} Gbytes")
                    t0 = time.time() if WORLD_MPI_RANK == 0 else None
        
        T = WORLD_MPI_COMM.reduce(sendobj=T_local, op=MPI.SUM, root=0)
        #cp.cuda.get_current_stream().synchronize()
        WORLD_MPI_COMM.Barrier()

        return T
    

    def atrg_tensor(self):
        dim = self.dim

    