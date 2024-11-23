# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any, Dict

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Dict[str, Any]) -> Fn:
    """Decorator to JIT compile a function for CUDA device execution.

    Args:
        fn: Function to compile
        kwargs: Additional arguments to pass to numba.cuda.jit

    Returns:
        JIT compiled function for CUDA device

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable, **kwargs: Dict[str, Any]) -> FakeCUDAKernel:
    """Decorator to JIT compile a function for CUDA execution.

    Args:
        fn: Function to compile
        kwargs: Additional arguments to pass to numba.cuda.jit

    Returns:
        JIT compiled CUDA kernel

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Applies a binary function element-wise across two tensors.

        Args:
            fn: Binary function to apply element-wise

        Returns:
            Function that takes two tensors and returns result tensor

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduces a tensor along a dimension using a binary function.

        Args:
            fn: Binary reduction function
            start: Initial value for reduction

        Returns:
            Function that takes tensor and dimension and returns reduced tensor

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication of two tensors.

        Args:
            a: First tensor
            b: Second tensor

        Returns:
            Result of matrix multiplication

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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        if i < out_size:
           
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            out[out_pos] = fn(in_storage[in_pos])




    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        if i < out_size:
           
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])



    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    # Shared memory for block-level reduction
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Global thread index
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # Thread index within the block
    thread_id = cuda.threadIdx.x

    # Load input data into shared memory
    if i < size:
        cache[thread_id] = a[i]
    else:
        cache[thread_id] = 0.0
    cuda.syncthreads()

    # Perform reduction in shared memory
    stride = 1
    while stride < BLOCK_DIM:
        if thread_id % (2 * stride) == 0 and thread_id + stride < BLOCK_DIM:
            cache[thread_id] += cache[thread_id + stride]
        cuda.syncthreads()
        stride *= 2

    # Write the result from thread 0 of each block to the output
    if thread_id == 0:
        out[cuda.blockIdx.x] = cache[0]

jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Perform practice sum reduction on tensor.

    Args:
        a: Input tensor

    Returns:
        Reduced tensor data

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        i = cuda.threadIdx.x
        block_id = cuda.blockIdx.x
        if block_id < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(block_id, out_shape, out_index)
            cache[i] = reduce_value

            for j in range(i, a_shape[reduce_dim], BLOCK_DIM):
                out_index[reduce_dim] = j
                a_pos = index_to_position(out_index, a_strides)
                cache[i] = fn(cache[i], a_storage[a_pos])
            cuda.syncthreads()

            stride = BLOCK_DIM // 2
            while stride > 0:
                if i < stride:
                    cache[i] = fn(cache[i], cache[i + stride])
                cuda.syncthreads()
                stride //= 2

            if i == 0:
                out_pos = index_to_position(out_index, out_strides)
                out[out_pos] = cache[0]




    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    # TODO: Implement for Task 3.3.
    TILE_SIZE = 32  # Fixed tile size for small matrices.

    # Shared memory for storing tiles of A and B
    tile_a = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)
    tile_b = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)

    # Thread and block indices
    thread_row = cuda.threadIdx.x
    thread_col = cuda.threadIdx.y

    # Compute global row and column indices in the output matrix
    global_row = cuda.blockIdx.x * TILE_SIZE + thread_row
    global_col = cuda.blockIdx.y * TILE_SIZE + thread_col

    # Initialize the local result for this thread
    local_result = 0.0

    # Iterate through tiles in the shared dimension
    for tile_offset in range(0, size, TILE_SIZE):
        # Load a tile of A into shared memory
        if global_row < size and tile_offset + thread_col < size:
            tile_a[thread_row, thread_col] = a[global_row * size + tile_offset + thread_col]
        else:
            tile_a[thread_row, thread_col] = 0.0

        # Load a tile of B into shared memory
        if global_col < size and tile_offset + thread_row < size:
            tile_b[thread_row, thread_col] = b[(tile_offset + thread_row) * size + global_col]
        else:
            tile_b[thread_row, thread_col] = 0.0

        # Synchronize to ensure all threads have loaded their data
        cuda.syncthreads()

        # Perform partial matrix multiplication for this tile
        for k in range(TILE_SIZE):
            local_result += tile_a[thread_row, k] * tile_b[k, thread_col]

        # Synchronize before loading the next tile
        cuda.syncthreads()

    # Write the computed value to the output matrix
    if global_row < size and global_col < size:
        out[global_row * size + global_col] = local_result




jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform practice matrix multiplication on tensors.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Result tensor data

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    TILE_SIZE = 32  # Define tile size for shared memory blocks.

    # Identify the current batch (for batch processing support)
    batch_idx = cuda.blockIdx.z

    # Thread indices within the current block
    local_row = cuda.threadIdx.x
    local_col = cuda.threadIdx.y

    # Compute global row and column indices in the output matrix
    global_row = cuda.blockIdx.x * TILE_SIZE + local_row
    global_col = cuda.blockIdx.y * TILE_SIZE + local_col

    # Shared memory allocation for tiles of A and B
    tile_a = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)
    tile_b = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)

    # Accumulator for the result of the current thread
    local_result = 0.0

    # Compute the number of tiles needed to cover the shared dimension
    num_tiles = (a_shape[-1] + TILE_SIZE - 1) // TILE_SIZE

    # Iterate through each tile in the shared dimension
    for tile_idx in range(num_tiles):
        # Determine the column in A and the row in B for this tile
        a_col = tile_idx * TILE_SIZE + local_col
        b_row = tile_idx * TILE_SIZE + local_row

        # Load a tile of A into shared memory
        if global_row < a_shape[-2] and a_col < a_shape[-1]:
            a_index = (
                batch_idx * (a_strides[0] if a_shape[0] > 1 else 0)
                + global_row * a_strides[1]
                + a_col * a_strides[2]
            )
            tile_a[local_row, local_col] = a_storage[a_index]
        else:
            tile_a[local_row, local_col] = 0.0

        # Load a tile of B into shared memory
        if b_row < b_shape[-2] and global_col < b_shape[-1]:
            b_index = (
                batch_idx * (b_strides[0] if b_shape[0] > 1 else 0)
                + b_row * b_strides[1]
                + global_col * b_strides[2]
            )
            tile_b[local_row, local_col] = b_storage[b_index]
        else:
            tile_b[local_row, local_col] = 0.0

        # Synchronize to ensure all threads have loaded their data
        cuda.syncthreads()

        # Perform the dot product for the current tile
        for k in range(TILE_SIZE):
            local_result += tile_a[local_row, k] * tile_b[k, local_col]

        # Synchronize before loading the next tile
        cuda.syncthreads()

    # Write the computed value to the output tensor
    if global_row < out_shape[-2] and global_col < out_shape[-1]:
        out_index = (
            batch_idx * out_strides[0]
            + global_row * out_strides[1]
            + global_col * out_strides[2]
        )
        out[out_index] = local_result



tensor_matrix_multiply = jit(_tensor_matrix_multiply)