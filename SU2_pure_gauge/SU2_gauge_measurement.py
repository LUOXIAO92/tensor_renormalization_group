from trg.hotrg.HOTRG import Tensor_HOTRG, HOTRG_2d, HOTRG_info
from trg.atrg.ATRG import Tensor_ATRG, ATRG_3d, ATRG_info

from .SU2_pure_gauge_init import SU2_pure_gauge

class two_dims_su2puregauge_hotrg(HOTRG_2d):
    def __init__(self, 
                 info:HOTRG_info,
                 dim,
                 Ks, 
                 β,
                 ε,
                 χinit,
                 init_tensor_chunk,
                 usegpu):
        
        super().__init__(info=info)
        
        self.dim  = dim
        self.Dcut = info.Dcut

        self.Ks    = Ks
        self.β     = β
        self.ε     = ε
        self.χinit = χinit
        self.init_tensor_chunk = init_tensor_chunk

        self.usegpu = usegpu

    def cal_free_energy(self):
        su2puregauge = SU2_pure_gauge(dim     = self.dim, 
                                      Dcut    = self.Dcut, 
                                      Ks      = self.Ks,
                                      β       = self.β, 
                                      ε       = self.ε, 
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
    

class three_dims_su2puregauge_atrg(ATRG_3d):
    def __init__(self, 
                 info:ATRG_info,
                 dim,
                 Ks, 
                 β,
                 ε,
                 χinit,
                 init_tensor_chunk,
                 usegpu):
        
        super().__init__(info=info)
        
        self.dim  = dim
        self.Dcut = info.Dcut

        self.Ks    = Ks
        self.β     = β
        self.ε     = ε
        self.χinit = χinit
        self.init_tensor_chunk = init_tensor_chunk

        self.usegpu = usegpu


    def cal_free_energy(self):
        su2puregauge = SU2_pure_gauge(dim     = self.dim, 
                                      Dcut    = self.Dcut, 
                                      Ks      = self.Ks,
                                      β       = self.β, 
                                      ε       = self.ε, 
                                      comm    = self.comm, 
                                      use_gpu = self.usegpu)
        
        #P = su2puregauge.plaquette_tensor(chi = self.χinit, 
        #                                  chunk = self.init_tensor_chunk, 
        #                                  legs_to_hosvd = [0])
        
        chunk_environment = (1,)
        Tinit = su2puregauge.atrg_tensor(self.Dcut, 
                                         self.Dcut, 
                                         self.init_tensor_chunk, 
                                         chunk_environment,
                                         legs_to_hosvd=[0, 1],
                                         truncate_eps=1e-10,
                                         degeneracy_eps=0,
                                         verbose=True)