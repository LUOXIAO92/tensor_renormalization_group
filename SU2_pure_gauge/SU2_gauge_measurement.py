#from ..trg.hotrg.HOTRG import Tensor_HOTRG, HOTRG_2d, HOTRG_info
from trg.hotrg.HOTRG import Tensor_HOTRG, HOTRG_2d, HOTRG_info
from .SU2_pure_gauge_init import SU2_pure_gauge

class two_dims_su2puregauge_hotrg(HOTRG_2d):
    def __init__(self, 
                 info:HOTRG_info,
                 dim,
                 Ks, 
                 beta, 
                 epsilon, 
                 χinit,
                 init_tensor_chunk,
                 usegpu):
        
        super().__init__(info=info)
        
        #self.info = info
        self.dim  = dim
        self.Dcut = info.Dcut

        self.Ks      = Ks
        self.beta    = beta
        self.epsilon = epsilon
        self.χinit   = χinit
        self.init_tensor_chunk = init_tensor_chunk

        self.usegpu = usegpu

    def cal_free_energy(self):
        #Kt, Ka, Kb = self.Ks
        su2puregauge = SU2_pure_gauge(dim     = self.dim, 
                                      Dcut    = self.Dcut, 
                                      Ks      = self.Ks,
                                      beta    = self.beta, 
                                      epsilon = self.epsilon, 
                                      comm    = self.comm, 
                                      use_gpu = self.usegpu)
        
        Tinit = su2puregauge.plaquette_tensor(chi   = self.χinit, 
                                              chunk = self.init_tensor_chunk, 
                                              legs_to_hosvd = [0])
        T = Tensor_HOTRG(dim  = self.dim, 
                         Dcut = self.Dcut, 
                         comm = self.comm)
        T = T.initialize(Tinit)
        
        T = self.pure_tensor_renormalization(T)
        lnZoV = self.cal_lnZoverV(T)
        
        return lnZoV
    
