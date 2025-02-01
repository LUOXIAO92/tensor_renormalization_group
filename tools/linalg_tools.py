try:
    import cupy as cp
except:
    pass
import numpy as np

from functools import reduce
from operator import mul
import copy

def truncate(s0, k:int, degeneracy_eps, truncate_eps):
    """
    Truncation for svd or eigh with given singular/eigen values and rank k. 

    Returns truncated rank χ, in which 1 < χ <= k.
    
    Parameters
    ----------
    s0 : Singular/eigen values 
    k : Truncation rank
    degeneracy_eps : We can say s[i+1] and s[i] are the same degeneracy level 
                     when |s[i+1] - s0[i]| / s[i+1] < degeneracy_eps 
    truncate_eps: Truncation tolerance
    
    Returns
    -------
    χ : truncated bond dimension, in which 1 < χ <= k
    """
    if type(s0) == np.ndarray:
        xp = np
    elif type(s0) == cp.ndarray:
        xp = cp
    else:
        raise SystemExit("Only support numpy.ndarray or cupy.ndarray")

    if degeneracy_eps == 0 and truncate_eps == 0:
        return len(s0), 0
    if k > len(s0):
        return len(s0), 0
    
    s = s0.copy()
    smax = copy.copy(s[0])
    s /= smax

    #truncate bond deminsion
    diff = xp.zeros(len(s)+1)
    diff[1:len(s)] = xp.abs(s[1:]-s[:len(s)-1]) / s[1:]
    diff[len(s)] = 1.0
    delta = xp.where(diff > degeneracy_eps, 1, 0)

    #find maximum of χ1 to avoid degeneracy, which 0 < χ1 <= k.
    for χ1 in range(k, 0, -1):
        if delta[χ1] == 1:
            break

    #truncate tolerance
    s_sqr = s**2
    norms_sqr = xp.sum(s_sqr)
    errors = xp.asarray([ xp.sqrt(xp.sum(s_sqr[i:]/norms_sqr)) for i in range(len(s0)) ])
    errors_chi1 = errors[:χ1]
    χ2 = len(errors_chi1[errors_chi1 > truncate_eps])

    #find maximum of χ to avoid degeneracy, which χ2 <= χ <= χ1 <= k.
    if χ2 == χ1:
        χ = χ1
    elif χ2 < χ1:
        for χ in range(χ2, χ1+1):
            if delta[χ] == 1:
                break
    else:
        raise ValueError("χ2 should be smaller than or equal χ1")
    
    if len(errors) == χ:
        error = 0.0
    elif len(errors) > χ:
        error = errors[χ]
    
    return χ, error

def svd(A, shape, k, truncate_eps=0, degeneracy_eps=0, split=False, return_err=False):
    """
    Svd for a tensor A with a given shape. 

    Returns `u`, `s`, `vh`, `error` or `us`, `svh`, `s`, `error`
    
    Parameters
    ----------
    `A` : Tensor
    `shape` : Reshape tensor to matrix with shape.    
    `k` : Truncation rank
    `degeneracy_eps` : We can say s[i+1] and s[i] are the same degeneracy level 
                     when |s[i+1] - s0[i]| / s[i+1] < degeneracy_eps 
    `truncate_eps` : Truncation tolerance
    `split` : True if split A

    Returns
    -------
    `u`, `s`, `vh`, `error` : if `split`=False, A = u @ s @ vh
    `us`, `svh`, `s`, `error` : if `split`=True, A = us @ svh, in which us = sum_n u[m,n]*s[n], svh = sum_m s[m]*vh[m,n]

    Example
    -------
    For a given 5-rank tensor A_{i,j,k,l,m}, we want to compute a svd for A_{ilm,kj}. 

    So the parameter `shape` should be shape=[[0,3,4],[2,1]], in order to reshape the input tensor A_{i,j,k,l,m} to A_{i,l,m,k,j}, 
    and group the legs {ilm} and {kj}.
    """
    if type(A) == np.ndarray:
        xp = np
    elif type(A) == cp.ndarray:
        xp = cp
    else:
        raise SystemExit("Only support numpy.ndarray or cupy.ndarray")
   
    llen = len(shape[0])
    rlen = len(shape[1])

    A_newaxis   = tuple(shape[0]+shape[1])
    A_newshape  = tuple([A.shape[i] for i in A_newaxis])
    A_mat_shape = (reduce(mul, A_newshape[:llen]), reduce(mul, A_newshape[llen:]))

    #transpose A to new axis, reshape it to matrix, and then svd it
    A = xp.transpose(A, A_newaxis)
    A = xp.reshape(A, A_mat_shape)
    u, s, vh = xp.linalg.svd(A, full_matrices=False)

    #reshape matrix A to tensor, then restore the axis
    A = xp.reshape(A, A_newshape)
    A_newaxis = xp.asarray(A_newaxis)
    restoreaxis = tuple(xp.argsort(A_newaxis).tolist())
    A = xp.transpose(A, restoreaxis)


    χ, error = truncate(s, k, degeneracy_eps, truncate_eps)
    
    u = u[:,:χ]
    s = s[:χ]
    vh = vh[:χ,:]

    if split:
        us  = xp.einsum("ai,i->ai", u,  xp.sqrt(s))
        svh = xp.einsum("ib,i->ib", vh, xp.sqrt(s))
        us  = xp.reshape(us,  A_newshape[:llen]+(χ,))
        svh = xp.reshape(svh, (χ,)+A_newshape[llen:])
        if return_err:
            return us, svh, s, error
        else:
            return us, svh, s
    else:
        u  = xp.reshape(u,  A_newshape[:llen]+(χ,))
        vh = xp.reshape(vh, (χ,)+A_newshape[llen:])
        if return_err:
            return u, s, vh, error
        else:
            return u, s, vh

def eigh(A, shape, k, truncate_eps=0, degeneracy_eps=5e-2, return_err=False):
    """
    Eigh for a tensor A with a given shape. 

    Returns `e`, `u`
    
    Parameters
    ----------
    A : Tensor
    shape : Reshape tensor to matrix with 'shape'.    
    k : Truncation rank
    degeneracy_eps : We can say s[i+1] and s[i] are the same degeneracy level 
                     when |s[i+1] - s0[i]| / s[i+1] < degeneracy_eps 
    truncate_eps : Truncation tolerance

    Returns
    -------
    e, u : A = u @ e @ uh

    Example
    -------
    For a given 4-rank tensor A_{i,j,k,l}, we want to compute a svd for A_{il,kj}. 
    
    So the parameter 'shape' should be shape=[[0,3],[2,1]], inorder to reshape the input tensor A_{i,j,k,l} to A_{i,l,k,j}, 
    and group the legs {il} and {kj}.
    """

    if type(A) == np.ndarray:
        xp = np
    elif type(A) == cp.ndarray:
        xp = cp
    else:
        raise SystemExit("Only support numpy.ndarray or cupy.ndarray")

    llen = len(shape[0])
    rlen = len(shape[1])

    A_newaxis   = tuple(shape[0]+shape[1])
    A_newshape  = tuple([A.shape[i] for i in A_newaxis])
    A_mat_shape = (reduce(mul, A_newshape[:llen]), reduce(mul, A_newshape[llen:]))

    #transpose A to new axis, reshape it to matrix, and then eigh it
    A = xp.transpose(A, A_newaxis)
    A = xp.reshape(A, A_mat_shape)
    e, u = xp.linalg.eigh(A)
    e = xp.abs(e)
    e = e[::-1]
    u = u[:,::-1]

    #reshape matrix A to tensor, then restore the axis
    A = xp.reshape(A, A_newshape)
    A_newaxis = xp.asarray(A_newaxis)
    restoreaxis = tuple(xp.argsort(A_newaxis).tolist())
    A = xp.transpose(A, restoreaxis)

    χ, error = truncate(e, k, degeneracy_eps, truncate_eps)

    u = u[:,:χ]
    e = e[:χ]
    u  = xp.reshape(u,  A_newshape[:llen]+(χ,))

    if return_err:
        return e, u, error
    else:
        return e, u
