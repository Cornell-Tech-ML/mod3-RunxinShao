from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function with Numba, inlining always."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))
        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations
def tensor_map(fn: Callable[[float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
       
        dims_match = len(out_shape) == len(in_shape)
        shapes_match = True
        if dims_match:
            for i in range(len(out_shape)):
                if out_shape[i] != in_shape[i]:
                    shapes_match = False
                    break
        
       
        strides_match = True
        if dims_match:
            for i in range(len(out_strides)):
                if out_strides[i] != in_strides[i]:
                    strides_match = False
                    break
                    
        if dims_match and shapes_match and strides_match:
          
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            
            for i in prange(len(out)):
                out_idx = np.zeros(MAX_DIMS, dtype=np.int32)
                in_idx = np.zeros(MAX_DIMS, dtype=np.int32)
                to_index(i, out_shape, out_idx)
                broadcast_index(out_idx, out_shape, in_shape, in_idx)
                out_pos = index_to_position(out_idx, out_strides)
                in_pos = index_to_position(in_idx, in_strides)
                out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)

def tensor_zip(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
       
        dims_match = (len(out_shape) == len(a_shape) == len(b_shape))
        shapes_match = True
        if dims_match:
            for i in range(len(out_shape)):
                if not (out_shape[i] == a_shape[i] == b_shape[i]):
                    shapes_match = False
                    break
                    
        
        strides_match = True
        if dims_match:
            for i in range(len(out_strides)):
                if not (out_strides[i] == a_strides[i] == b_strides[i]):
                    strides_match = False
                    break

        if dims_match and shapes_match and strides_match:
           
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
           
            for i in prange(len(out)):
                out_idx = np.zeros(MAX_DIMS, dtype=np.int32)
                a_idx = np.zeros(MAX_DIMS, dtype=np.int32)
                b_idx = np.zeros(MAX_DIMS, dtype=np.int32)
                
                to_index(i, out_shape, out_idx)
                broadcast_index(out_idx, out_shape, a_shape, a_idx)
                broadcast_index(out_idx, out_shape, b_shape, b_idx)
                
                out_pos = index_to_position(out_idx, out_strides)
                a_pos = index_to_position(a_idx, a_strides)
                b_pos = index_to_position(b_idx, b_strides)
                
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)

def tensor_reduce(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        
        for i in prange(len(out)):
            out_idx = np.zeros(MAX_DIMS, dtype=np.int32)
            to_index(i, out_shape, out_idx)
            out_pos = index_to_position(out_idx, out_strides)
            
            dim_size = a_shape[reduce_dim]
            dim_stride = a_strides[reduce_dim]
            
            result = out[out_pos]
            
            base_pos = index_to_position(out_idx, a_strides)
           
            curr_pos = base_pos
            for j in range(dim_size):
                result = fn(result, a_storage[curr_pos])
                curr_pos += dim_stride
                
            out[out_pos] = result

    return njit(_reduce, parallel=True)



from numba import njit, prange
from .tensor_data import Storage, Shape, Strides

@njit(parallel=True)
def _tensor_matrix_multiply(
    out_storage: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
   
    batch_size = out_shape[0] if len(out_shape) > 2 else 1
    M = out_shape[-2]
    N = out_shape[-1]
    K = a_shape[-1]  

    a_batch_stride = a_strides[0] if len(a_shape) > 2 and a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if len(b_shape) > 2 and b_shape[0] > 1 else 0
    out_batch_stride = out_strides[0] if len(out_shape) > 2 else 0

   
    for batch in prange(batch_size):
        
        for i in range(M):
            for j in range(N):
                
                result = 0.0
                
                for k in range(K):
                   
                    a_idx = (
                        batch * a_batch_stride +
                        i * a_strides[-2] +
                        k * a_strides[-1]
                    )
                    b_idx = (
                        batch * b_batch_stride +
                        k * b_strides[-2] +
                        j * b_strides[-1]
                    )
                  
                    result += a_storage[a_idx] * b_storage[b_idx]
             
                out_idx = (
                    batch * out_batch_stride +
                    i * out_strides[-2] +
                    j * out_strides[-1]
                )
           
                out_storage[out_idx] = result

tensor_matrix_multiply = _tensor_matrix_multiply