import os
import sys

from mpi4py import MPI 
from tools.mpi_tools import gpu_syn
import math
import opt_einsum as oe

try:
    import cupy as cp
except:
    pass
import numpy as np

class HOTRG_info:
    def __init__(self,
                 Dcut:int,
                 rgiter:str|list,
                 truncate_eps:float,
                 degeneracy_eps:float, 
                 gilt_eps:float, 
                 Ngilt:int, 
                 Ncutlegs:int,
                 reduced_matrix_chunk:tuple, 
                 coarse_graining_chunk:tuple,
                 verbose:bool,
                 save_details:bool, 
                 outdir:str,
                 comm:MPI.Intercomm):
        
        #Chunking information
        self.reduced_matrix_chunk  = reduced_matrix_chunk
        self.coarse_graining_chunk = coarse_graining_chunk

        #Tolerance
        self.truncate_eps   = truncate_eps
        self.degeneracy_eps = degeneracy_eps

        #HOTRG runtime info
        self.Dcut         = Dcut
        self.rgiter       = rgiter
        self.gilt_eps     = gilt_eps
        self.Ngilt        = Ngilt
        self.Ncutlegs     = Ncutlegs
        self.save_details = save_details
        self.verbose      = verbose
        self.outdir       = outdir

        #MPI commucator
        self.comm = comm

class Tensor_HOTRG:
    def __init__(self, dim, Dcut, comm:MPI.Intercomm):
        #basic data----------
        self.dim    = dim
        self.Dcut   = Dcut
        self.comm   = comm

        #need to be initialized---
        self.is_impure = False
        self.T      = None
        self.coor   = None
        self.rgstep    = {}
        self.nrgsteps  = 0
        self.factor    = {}
        self.ln_factor = {}

        self.xp = np
        self.usegpu  = False
        self.backend = None
        self.initialized = False

        self.shape  = (0,)
        self.ndim   = 0
        self.nbytes = 0
        self.size   = 0

        #The rank that this tensor exist
        self.where = None
        
    def initialize(self, T, is_impure=False, coor=None, root=0):
        if is_impure:
            assert coor is not None, "Coordinate of impure tensor is needed."
        self.is_impure = is_impure

        self.T = T
        if self.comm.Get_rank() == root:
            self.where = root
            if type(T) == np.ndarray:
                self.usegpu = False
            elif type(T) == cp.ndarray:
                self.usegpu = True
            else:
                raise TypeError('Only support numpy.ndarray and cupy.ndarray')
            
            self.backend = type(T)
            self.shape  = T.shape
            self.ndim   = T.ndim
            self.nbytes = T.nbytes
            self.size   = T.size

        self.shape   = self.comm.bcast(obj=self.shape  , root=root)
        self.ndim    = self.comm.bcast(obj=self.ndim   , root=root)
        self.nbytes  = self.comm.bcast(obj=self.nbytes , root=root)
        self.size    = self.comm.bcast(obj=self.size   , root=root)
        self.where   = self.comm.bcast(obj=self.where  , root=root)
        self.backend = self.comm.bcast(obj=self.backend, root=root)

        if self.backend == np.ndarray:
            self.xp = np
        elif self.backend == cp.ndarray:
            self.xp = cp

        if self.dim   == 2:
            self.rgstep = {'X':0, 'Y':0}
        elif self.dim == 3:
            self.rgstep = {'X':0, 'Y':0, 'T':0}
        elif self.dim == 4:
            self.rgstep = {'X':0, 'Y':0, 'Z':0, 'T':0}
        else:
            raise AssertionError('Only support 2,3,4 dimensional system.')
        
        self.initialized = True
        gpu_syn(self.usegpu)
        self.comm.barrier()
        
        return self
        
    def trace(self):
        if self.dim == 2:
            subscripts = 'xxyy'
        elif self.dim == 3:
            subscripts = 'xxyytt'
        elif self.dim == 4:
            subscripts = 'xxyyzztt'

        comm = self.comm
        rank = comm.Get_rank()
        if rank == self.where:
            tr = oe.contract(subscripts, self.T)
        else:
            tr = None
        tr = comm.bcast(obj=tr, root=self.where)
        gpu_syn(self.usegpu)
        comm.barrier()

        return tr
    
    def update_nrgsteps(self, direction:str):
        assert direction in self.rgstep.keys()
        self.rgstep[direction] += 1
        self.nrgsteps = sum(self.rgstep.values())
    
    def normalize(self):
        
        rank = self.comm.Get_rank()
        c = self.trace()
        if rank == self.where:
            c = self.xp.abs(c)
            self.T /= c
        c = self.comm.bcast(obj=c, root=self.where)
        gpu_syn(self.usegpu)
        self.comm.barrier()

        self.factor[self.nrgsteps] = c
        self.ln_factor[self.nrgsteps] = self.xp.log(c) / 2**(self.nrgsteps)
        
        return self
    
    def move_to_rank(self, dest_rank):
        comm = self.comm
        rank = comm.Get_rank()
        if rank == self.where:
            comm.send(obj=self.T, dest=dest_rank, tag=dest_rank)
        else:
            self.T = comm.recv(source=self.where, tag=dest_rank)
        gpu_syn(self.usegpu)
        comm.barrier()

        
        gpu_syn(self.usegpu)
        comm.barrier()
        self.where = dest_rank
        if rank != dest_rank:
            self.T = None
        else:
            assert self.T is not None
        gpu_syn(self.usegpu)
        comm.barrier()

        return self
    
    def to_device(self, device):
        """
        device : 'gpu' or 'cpu'
        """

        if self.comm.Get_rank() == self.where:
            if device == 'cpu':
                self.T = np.asarray(self.T)
                self.backend = np.ndarray
                self.usegpu = False
            elif device == 'gpu':
                self.T = cp.asarray(self.T)
                self.backend = cp.ndarray
                self.usegpu = True

        self.backend = self.comm.bcast(obj=self.backend, root=self.where)

        if self.backend == np.ndarray:
            self.xp = np
        elif self.backend == cp.ndarray:
            self.xp = cp

        gpu_syn(self.usegpu)
        self.comm.Get_rank()

        return self


class HOTRG_2d:
    def __init__(self, info:HOTRG_info):
        
        self.info = info
        self.comm = info.comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def transfer_matrix_eigenvalues(self, Tpure:Tensor_HOTRG):
        pass

    def save_singular_values(self, Tpure:Tensor_HOTRG):

        if self.comm.Get_rank() == Tpure.where:
            if self.info.save_details:
                mat = Tpure.T.transpose((0,2,1,3))
                mat = mat.reshape((mat.shape[0]*mat.shape[1], mat.shape[2]*mat.shape[3]))
                xp = Tpure.xp

                _, s, _ = xp.linalg.svd(mat)

                nrgsteps = sum(Tpure.rgstep.values())
                if nrgsteps == 0:
                    mode = 'w'
                else:
                    mode = 'a'
                output_dir = self.info.outdir.rstrip('/') + '/tensor'
                filename = output_dir + f'/tensor_n{nrgsteps}.dat'
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                with open(filename, mode) as out:
                    s0 = xp.max(s)
                    s = s / s0
                    out.write(f'#lambda_0={s0:.12e}\n')
                    for ss in s:
                        out.write(f'{ss:.12e}\n')
        
        gpu_syn(Tpure.usegpu)
        self.comm.barrier()

    def cal_lnZoverV(self, Tpure:Tensor_HOTRG):
        xp = Tpure.xp
        TrT = Tpure.trace()
        lnnorm = sum(Tpure.ln_factor.values())
        lnZoV = lnnorm + xp.log(TrT) / 2**(Tpure.nrgsteps)

        if self.info.save_details:
            if self.comm.Get_rank() == Tpure.where:
                if Tpure.nrgsteps == 0:
                    mode = 'w'
                else:
                    mode = 'a'
                
                outdir = self.info.outdir
                fname  = outdir.rstrip('/') + '/lnZoverV.dat'

                with open(fname, mode) as out:
                    out.write(f'{lnZoV.real:.12e}\n')

        return lnZoV

    def pure_tensor_renormalization(self, Tpure:Tensor_HOTRG):
        from .HOTRG_2d_core import new_pure_tensor

        def exec_rg(Tpure:Tensor_HOTRG, direction):
            Tpure = new_pure_tensor(self.info, Tpure, direction)

            Tpure = Tpure.normalize()
            nrgsteps = Tpure.nrgsteps
            Tpure.ln_factor[nrgsteps] = math.log(Tpure.factor[nrgsteps]) / 2**(nrgsteps)
            
            gpu_syn(Tpure.usegpu)
            self.comm.barrier()

            return Tpure
        
        Tpure = Tpure.normalize()
        nrgsteps = Tpure.nrgsteps
        Tpure.ln_factor[nrgsteps] = math.log(Tpure.factor[nrgsteps]) / 2**(nrgsteps)
        self.save_singular_values(Tpure)
        lnZoV = self.cal_lnZoverV(Tpure)

        if self.rank == 0:
            print(f"lnZ/V={lnZoV}. c={Tpure.factor[Tpure.nrgsteps]}")

        gpu_syn(Tpure.usegpu)
        self.comm.barrier()
        
        for direction in self.info.rgiter:
            Tpure.update_nrgsteps(direction)
            if self.rank == 0:
                print(f"start {Tpure.nrgsteps}-th rgstep. Coarse graining direction is {direction}")
            
            Tpure = exec_rg(Tpure, direction)

            #compute operators
            self.save_singular_values(Tpure)
            lnZoV = self.cal_lnZoverV(Tpure)

            if self.rank == 0:
                print(f"{Tpure.nrgsteps}-th rgstep finished. lnZ/V={lnZoV}. c={Tpure.factor[Tpure.nrgsteps]}")
            

        return Tpure
    
    def one_point_function_renormalization(self, Tpure:Tensor_HOTRG, *Timpure:Tensor_HOTRG):
        pass

    def two_point_function_renormalization(self, Tpure:Tensor_HOTRG, *Timpure:Tensor_HOTRG):
        pass



