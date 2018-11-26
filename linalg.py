# import numpy as np
import numpy
import numpy as np
try:
       import cupy as cp
except:
       ...
from copy import deepcopy
# from scipy.sparse import block_diag, bmat, diags, vstack, lil_matrix, linalg, issparse
from scipy.stats import rankdata

def percentile_cupy(a, q, axis=None, out=None, interpolation='linear',
                              keepdims=False):
       """Computes the q-th percentile of the data along the specified axis.
       Args:
        a (cupy.ndarray): Array for which to compute percentiles.
        q (float, tuple of floats or cupy.ndarray): Percentiles to compute
            in the range between 0 and 100 inclusive.
        axis (int or tuple of ints): Along which axis or axes to compute the
            percentiles. The flattened array is used by default.
        out (cupy.ndarray): Output array.
        interpolation (str): Interpolation method when a quantile lies between
            two data points. ``linear`` interpolation is used by default.
            Supported interpolations are``lower``, ``higher``, ``midpoint``,
            ``nearest`` and ``linear``.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.
       Returns:
        cupy.ndarray: The percentiles of ``a``, along the axis if specified.
       .. seealso:: :func:`numpy.percentile`
       """
       q = cp.asarray(q, dtype=a.dtype)
       if q.ndim == 0:
              q = q[None]
              zerod = True
       else:
              zerod = False
       if q.ndim > 1:
              raise ValueError('Expected q to have a dimension of 1.\n'
                               'Actual: {0} != 1'.format(q.ndim))
                  
       if keepdims:
              if axis is None:
                     keepdim = (1,) * a.ndim
              else:
                     keepdim = list(a.shape)
                     for ax in axis:
                            keepdim[ax % a.ndim] = 1
                     keepdim = tuple(keepdim)
       
       # Copy a since we need it sorted but without modifying the original array
       if isinstance(axis, int):
              axis = axis,
       if axis is None:
              ap = a.flatten()
              nkeep = 0
       else:
              # Reduce axes from a and put them last
              axis = tuple(ax % a.ndim for ax in axis)
              keep = set(range(a.ndim)) - set(axis)
              nkeep = len(keep)
              for i, s in enumerate(sorted(keep)):
                     a = a.swapaxes(i, s)
              ap = a.reshape(a.shape[:nkeep] + (-1,)).copy()
                                                 
       axis = -1
       ap.sort(axis=axis)
       Nx = ap.shape[axis]
       indices = q * 0.01 * (Nx - 1.)  # percents to decimals

       
       if interpolation == 'lower':
              indices = cp.floor(indices).astype(cp.int32)
       elif interpolation == 'higher':
              indices = cp.ceil(indices).astype(cp.int32)
       elif interpolation == 'midpoint':
              indices = 0.5 * (cp.floor(indices) + cp.ceil(indices))
       elif interpolation == 'nearest':
              # TODO(hvy): Implement nearest using around
              raise ValueError("'nearest' interpolation is not yet supported. "
                               'Please use any other interpolation method.')
       elif interpolation == 'linear':
              pass
       else:
              raise ValueError('Unexpected interpolation method.\n'
                               "Actual: '{0}' not in ('linear', 'lower', 'higher', "
                               "'midpoint')".format(interpolation))
                                                 
       if indices.dtype == cp.int32:
              ret = cp.rollaxis(ap, axis)
              ret = ret.take(indices, axis=0, out=out)
       else:
              if out is None:
                     ret = cp.empty(ap.shape[:-1] + q.shape, dtype=cp.float64)
              else:
                     ret = cp.rollaxis(out, 0, out.ndim)
              
              cp.ElementwiseKernel(
                     'S idx, raw T a, raw int32 offset', 'U ret',
                     '''
                     ptrdiff_t idx_below = floor(idx);
                     U weight_above = idx - idx_below;
                     ptrdiff_t offset_i = _ind.get()[0] * offset;
                     ret = a[offset_i + idx_below] * (1.0 - weight_above)
                     + a[offset_i + idx_below + 1] * weight_above;
                     ''',
                     'percentile_weightnening'
              )(indices, ap, ap.shape[-1] if ap.ndim > 1 else 0, ret)
              ret = cp.rollaxis(ret, -1)  # Roll q dimension back to first axis
                     
       if zerod:
              ret = ret.squeeze(0)
       if keepdims:
              if q.size > 1:
                     keepdim = (-1,) + keepdim
              ret = ret.reshape(keepdim)
                                          
       return cp.ascontiguousarray(ret)
                                                                             
def maxpercentile (arr, percentile, xp=np):
       '''keeps those elements of arr that are larger than the \"percentile\"-th percentile of values, fills the rest with 0s'''
       if xp==np:
              p = np.percentile (abs (arr), 100-percentile)
              
       else:
              p = percentile_cupy (abs (arr), 100-percentile)
              
       return xp.where ((abs (arr) > p), arr, xp.zeros (arr.shape))
       
def rank_maxpercentile (inarr, percentile, xp=np):
       arr = xp.ravel (inarr)
       p = xp.percentile (abs (arr), 100-percentile)
       maxvalues = arr [xp.where (abs (arr)>p)] 
       ranks = rankdata (maxvalues)
       indices = xp.where ((abs (arr) > p), True, False)
       outarr = maxpercentile (abs (arr) , percentile)
       outarr [indices]=ranks
       return outarr

def rank_minpercentile (inarr, percentile, xp=np):
       arr = xp.ravel (inarr)
       p = xp.percentile (abs (arr), percentile)
       minvalues = arr [xp.where (abs (arr)<p)] #also p%-ba eso ertekek 
       ranks = rankdata (-minvalues)
       indices = xp.where ((abs (arr) < p), True, False)
       outarr = xp.where ((abs (arr) < p), arr, xp.zeros (arr.shape))
       outarr [indices]=ranks
       return outarr

######################################


def newton_forward_step (A, Xk, xp=np):
    """A single step of the generalised Newton iteration"""
    Xk = xp.matrix (Xk)
    I = xp.matrix (xp.eye (len (A))) 
    # return np.matmul (Xk, 2*I-np.matmul (A, Xk)) 
    return Xk*(2*I-A*Xk)

def gen_newton_inv (A, X0=None, max_it=10, xp=np):
    """Matrix inversion using the generalised Newton method. Choice of the constant factor is based on http://www4.ncsu.edu/~aalexan3/articles/mat-inv-rep.pdf
    
    Args:
         A: matrix to invert
         X0: initial matrix of the iteration
         max_it: number of iterations to be done"""

    A = xp.matrix (A)
    
    if not xp.all (X0):
        constant = 2/xp.max(xp.matmul(xp.abs(A*xp.transpose (A)), xp.ones((len (A),1))))
        X0 = constant * xp.transpose(A)

    X=X0

    for i in range (max_it):
        X = newton_forward_step (A, X)

    return X

# A = np.array([[1,2,3],[0,1,4],[5,6,0]])
# print (gen_newton_inv(A, max_it=30)) 
# A = np.random.randn(3,3)
# INV = gen_newton_inv(A, max_it=100)
# print(A*INV)

# start_time = time.time()
# gen_newton_inv(A, max_it=100)
# print("--- %s seconds ---" % (time.time() - start_time))

def gd(M, d):
    if issparse (M):
        u, s, v = linalg.svds(M, k=d)
        D = xp.maximum(0, s)
        U = v
    else:
        u, s, v = xp.linalg.svd(M)
        D = xp.maximum(0, s[:d])
        U = v[:d]
    return xp.matmul(xp.diag(xp.sqrt(D)), U)

def dotproductrepr(A, d, tol=0):
    """Creates a d-dimensional dot product representation of a weighted graph with adjacency matrix A.

    Args:
    A: nxn adjacency matrix. Has to be symmetric.
    d: dimension of the resulting representation

    Output:
    X: vectors corresponding to the vertices are columns of X."""

    n = A.shape[0]
    if issparse(A):
        D = lil_matrix((n,n))
    else:
        D = xp.zeros((n,n), dtype = np.float32)
        
    change = tol+1
    while change > tol:
        X = gd(A+D, d)
        D_uj = xp.diag(np.array([xp.matmul(X[:,i].T, X[:,i]) for i in range(n)]))
        change = xp.linalg.norm(D - D_uj)
        print(change)
        D = deepcopy(D_uj)
        print(X[0][0])
    return X

    
#########################

def grads_to_mx_horizontal(gradmx):
    """Args: gradmx: (n_i, n_{i+1}) ndarray"""
    return block_diag(gradmx.T)

def grads_to_mx_vertical(gradmx):
    return vstack([diags(row) for row in gradmx.T])

def mlp_grads_to_adj_single(gradmx):
    return block_diag((grads_to_mx_horizontal(gradmx), grads_to_mx_vertical(gradmx)))

def mlp_grads_to_adj(gradlist):
    n1 = gradlist[0].shape[1]
    n_last = gradlist[-1].shape[0]
    upper = block_diag([mlp_grads_to_adj_single(gradmx) for gradmx in gradlist])
    A = bmat([[lil_matrix((upper.shape[0], n1)), upper],[lil_matrix((n_last, n1)), None]]) 
    return A + A.T
    
