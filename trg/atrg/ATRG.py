from mpi4py import MPI
import opt_einsum as oe
from tools.mpi_tools import gpu_syn

class Tensor_ATRG:
    """
    This class is for atrg tensor.

    Leg arrangement of the tensor should be:
    >>> for 2 dimensional system T_{TXtx} = U_{TXi} s_i VH_{itx}
    >>> for 3 dimensional system T_{TXYtxy} = U_{TXYi} s_i VH_{itxy}
    >>> for 4 dimensional system T_{TXYZtxyz} = U_{TXYZi} s_i VH_{itxyz}
    dim: Dimension of the system.
    is_impure: True if this is a impure tensor. Otherwise False
    loc: Location of impure tensor. Must be dictionary type. It can be a empty dictionary if pure tensor
    >>> loc={"T":t, "X":x, "Y":y, "Z":z}

    """
    def __init__(self, dim:int, Dcut:int, comm:MPI.Intercomm, usegpu=False):
        
        self.Dcut = Dcut
        self.dim  = dim
        self.comm = comm
        self.usegpu = usegpu

        #need to be initialized---
        self.U  = None
        self.s  = None
        self.VH = None
        self.coor   = None
        self.rgstep = {}
        self.factor = {}
        self.ln_factor = {}
        self.is_impure = False

        self.initialized = False

        #The rank that this tensor exist
        self.where = None

    def initialize(self, U, s, VH, is_impure=False, coor=None):
        if is_impure:
            assert coor is not None, "Coordinate of impure tensor is needed."
        self.is_impure = is_impure

        self.U, self.s, self.VH = U, s, VH
        assert (type(U) == type(s)) and (type(U) == type(VH)), 'U, s, VH must be kept on the same rank.'
        if (self.U is not None) and (self.s is not None) and (self.VH is not None):
            self.where = self.comm.Get_rank()

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
        """
        calculate trace of this tensor
        """

        if self.comm.Get_rank() == self.where:
            if self.s.ndim == 1:
                if self.dim == 2:
                    TrT = oe.contract("txi,i,itx", self.U, self.s, self.VH)
                elif self.dim == 3:
                    TrT = oe.contract("txyi,i,itxy", self.U, self.s, self.VH)
                elif self.dim == 4:
                    TrT = oe.contract("txyzi,i,itxyz", self.U, self.s, self.VH)
            elif self.s.ndim == 2:
                if self.dim == 2:
                    TrT = oe.contract("yxi,ij,jyx", self.U, self.s, self.VH)
                elif self.dim == 3:
                    TrT = oe.contract("txyi,ij,jtxy", self.U, self.s, self.VH)
                elif self.dim == 4:
                    TrT = oe.contract("txyzi,ij,jtxyz", self.U, self.s, self.VH)
        else:
            TrT = None

        return TrT
    
    def cal_X(self):
        """
        returns X_t, X_x, X_y, X_z
        >>> X_x: X direction, X1 = TrT^2 / [ T_{t1,a,y1,z1,t1,b,y1,z1} T_{t2,b,y2,z2,t2,a,y2,z2} ] 
        >>> X_y: Y direction, X2 = TrT^2 / [ T_{t1,x1,a,z1,t1,x1,b,z1} T_{t2,x2,b,z2,t2,x2,a,z2} ] 
        >>> X_z: Z direction, X3 = TrT^2 / [ T_{t1,x1,y1,a,t1,x1,y1,b} T_{t2,x2,y2,b,t2,x2,y2,a} ] 
        >>> X_t: T direction, X4 = TrT^2 / [ T_{a,x1,y1,z1,b,x1,y1,z1} T_{b,x2,y2,z2,a,x2,y2,z2} ] 
        """
        TrT = self.trace()

        if self.dim == 2:
            TTx = oe.contract("txi,i,itX,TXj,j,jTx", self.U, self.s, self.VH, self.U, self.s, self.VH)
            TTt = oe.contract("txi,i,iTx,TXj,j,jtX", self.U, self.s, self.VH, self.U, self.s, self.VH)

            Xx = TrT**2 / TTx
            Xt = TrT**2 / TTt

            return Xt, Xx

        if self.dim == 3:
            TTx = oe.contract("tXyi,i,itxy,TxYj,j,jTXY", self.U, self.s, self.VH, self.U, self.s, self.VH)
            TTy = oe.contract("txYi,i,itxy,TXyj,j,jTXY", self.U, self.s, self.VH, self.U, self.s, self.VH)
            TTt = oe.contract("Txyi,i,itxy,tXYj,j,jTXY", self.U, self.s, self.VH, self.U, self.s, self.VH)

            #X1 = oe.contract("tXyi,i,itxy->Xx", self.U, self.s, self.VH)
            #X2 = oe.contract("txyi,i,itXy->xX", self.U, self.s, self.VH)
            #TTx = oe.contract("Xx,xX", X1, X2)
            #
            #Y1 = oe.contract("txYi,i,itxy->Yy", self.U, self.s, self.VH)
            #Y2 = oe.contract("txyi,i,itxY->yY", self.U, self.s, self.VH)
            #TTy = oe.contract("Yy,yY", Y1, Y2)
            #
            #T1 = oe.contract("Txyi,i,itxy->Tt", self.U, self.s, self.VH)
            #T2 = oe.contract("txyi,i,iTxy->tT", self.U, self.s, self.VH)
            #TTt = oe.contract("Tt,tT", T1, T2)

            Xx = TrT**2 / TTx
            Xy = TrT**2 / TTy
            Xt = TrT**2 / TTt

            return Xt, Xx, Xy
        
        if self.dim == 4:
            TTx = oe.contract("txyzi,i,itXyz,TXYZj,j,jTxYZ", self.U, self.s, self.VH, self.U, self.s, self.VH)
            TTy = oe.contract("txyzi,i,itxYz,TXYZj,j,jTXyZ", self.U, self.s, self.VH, self.U, self.s, self.VH)
            TTz = oe.contract("txyzi,i,itxyZ,TXYZj,j,jTXYz", self.U, self.s, self.VH, self.U, self.s, self.VH)
            TTt = oe.contract("txyzi,i,iTxyz,TXYZj,j,jtXYZ", self.U, self.s, self.VH, self.U, self.s, self.VH)

            Xx = TrT**2 / TTx
            Xy = TrT**2 / TTy
            Xz = TrT**2 / TTz
            Xt = TrT**2 / TTt

            return Xt, Xx, Xy, Xz
    
    def update(self, U, s, VH):
        self.U  = U
        self.s  = s
        self.VH = VH
        return self
    
    def normalize(self):
        if self.usegpu:
            from cupy import abs
        else:
            from numpy import abs
        
        rank = self.comm.Get_rank()
        c = self.trace()
        if rank == self.where:
            c = abs(c)
            self.s /= c
        c = self.comm.bcast(obj=c, root=self.where)
        gpu_syn(self.usegpu)
        self.comm.barrier()

        self.factor[sum(self.rgstep.values())] = c
        
        return self, c