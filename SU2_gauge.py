from mpi4py import MPI
import argparse
import ast
import sys
from tools.mpi_tools import use_gpu

pars = argparse.ArgumentParser()
pars.add_argument('--dcut'  , default=32, type=int)
pars.add_argument('--K'     , default=(6,6,6))
pars.add_argument('--rgiter', default='XYXYXYXYXYXY')
pars.add_argument('--L'     , default=None)
pars.add_argument('--beta'  , default=3.0, type=float)
pars.add_argument('--eps'   , default=0  , type=float)
pars.add_argument('--init_tensor_chunk'  , default=(6,1,1))

pars.add_argument('--rgscheme'      , default='hotrg')
pars.add_argument('--degeneracy_eps', default=0, type=float)
pars.add_argument('--truncate_eps'  , default=0, type=float)
pars.add_argument('--gilt_eps', default=0, type=float)
pars.add_argument('--Ngilt'   , default=1, type=int)
pars.add_argument('--Ncutlegs', default=2, type=int)

pars.add_argument('--reduced_matrix_chunk' , default=(4,4,4,4))
pars.add_argument('--coarse_graining_chunk', default=(1,1,1))

pars.add_argument('--out_dir'     , default='./')
pars.add_argument('--verbose'     , default=True, type=bool)
pars.add_argument('--save_details', default=True, type=bool)

if __name__ == "__main__":

    WORLD_COMM = MPI.COMM_WORLD
    WORLD_SIZE = WORLD_COMM.Get_size()
    WORLD_RANK = WORLD_COMM.Get_rank()
    COMM_INFO  = WORLD_COMM.Get_info()
    use_gpu(usegpu=True, comm=WORLD_COMM)

    args = pars.parse_args()

    Dcut = args.dcut
    Ks   = ast.literal_eval(args.K)
    beta = args.beta
    eps  = args.eps

    rgiter = args.rgiter
    if args.L is not None:
        rgiter = ''
        L = ast.literal_eval(args.L)
        assert len(L) == 2, "RG loops --L must be '(lx, ly)'"
        for i in range(L[0]+L[1]):
            if i % 2 == 0:
                rgiter += 'Y'
            else:
                rgiter += 'X'
        
    init_tensor_chunk = ast.literal_eval(args.init_tensor_chunk)
    assert len(init_tensor_chunk) == 3

    rgscheme             = args.rgscheme
    degeneracy_eps       = args.degeneracy_eps
    truncate_eps         = args.truncate_eps
    reduced_matrix_chunk = ast.literal_eval(args.reduced_matrix_chunk)
    coarse_graining_chunk = ast.literal_eval(args.coarse_graining_chunk)
    assert len(reduced_matrix_chunk) == 4
    assert len(coarse_graining_chunk) == 3

    gilt_eps = args.gilt_eps
    Ngilt    = args.Ngilt
    Ncutlegs = args.Ncutlegs

    out_dir      = args.out_dir
    verbose      = args.verbose
    save_details = args.save_details

    from SU2_pure_gauge.SU2_gauge_measurement import two_dims_su2puregauge_hotrg as SU2_2dHOTRG
    from trg.hotrg.HOTRG import HOTRG_info

    info = HOTRG_info(Dcut           = Dcut, 
                      rgiter         = rgiter,
                      degeneracy_eps = degeneracy_eps,
                      truncate_eps   = truncate_eps,
                      gilt_eps       = gilt_eps, 
                      Ngilt          = Ngilt, 
                      Ncutlegs       = Ncutlegs,
                      reduced_matrix_chunk  = reduced_matrix_chunk,
                      coarse_graining_chunk = coarse_graining_chunk, 
                      verbose      = verbose,
                      save_details = save_details,
                      outdir       = out_dir,
                      comm         = WORLD_COMM)
    
    eps = None if eps == 0 else eps
    su2_2dhotrg = SU2_2dHOTRG(info    = info,
                              dim     = 2, 
                              Ks      = Ks, 
                              beta    = beta, 
                              epsilon = eps, 
                              χinit   = Dcut, 
                              init_tensor_chunk = init_tensor_chunk, 
                              usegpu  = True)
    
    lnZoV = su2_2dhotrg.cal_free_energy()
