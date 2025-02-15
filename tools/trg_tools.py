import opt_einsum as oe
import numpy as np

from mpi4py import MPI


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

