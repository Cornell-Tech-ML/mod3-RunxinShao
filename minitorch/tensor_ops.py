from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Type
from typing_extensions import Protocol
import numpy as np

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    """Protocol for a callable mapping function."""

    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function."""
        ...


class TensorOps:
    """Tensor operations including map, zip, and reduce."""

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order map function.

        Args:
        ----
            fn: A function to apply element-wise.

        Returns:
        -------
            A MapProto that applies the function `fn`.

        """
        ...

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Higher-order zip function.

        Args:
        ----
            fn: A function to apply pair-wise between two tensors.

        Returns:
        -------
            A function that applies `fn` between elements of two tensors.

        """
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Higher-order reduce function.

        Args:
        ----
            fn: A reduction function to apply.
            start: The initial value for the reduction.

        Returns:
        -------
            A function that reduces a tensor along a dimension.

        """
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication between two tensors.

        Args:
        ----
            a: The first input tensor.
            b: The second input tensor.

        Returns:
        -------
            A tensor resulting from the matrix multiplication of `a` and `b`.

        """
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    """Backend for tensor operations using a specific `TensorOps` class."""

    def __init__(self, ops: Type[TensorOps]):
        """Construct a tensor backend.

        Args:
        ----
            ops: An object implementing map, zip, and reduce functions.

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)
        self.lt_zip = ops.zip(operators.lt)
        self.gt_zip = ops.zip(operators.gt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    """Simple operations for tensor manipulation."""

    @staticmethod
    def sum_reduce(a: Tensor, dim: int) -> Tensor:
        """Reduce a tensor by summing along a specific dimension.

        Args:
        ----
            a: The input tensor to reduce.
            dim: The dimension to reduce along.

        Returns:
        -------
            A tensor with the reduced dimension.

        """
        f = tensor_reduce(operators.add)
        out_shape = list(a.shape)
        out_shape[dim] = 1

        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = 0.0

        f(*out.tuple(), *a.tuple(), dim)
        return out

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function.

        Args:
        ----
            fn: Function to apply element-wise to a tensor.

        Returns:
        -------
            A MapProto function to apply `fn`.

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Higher-order tensor zip function.

        Args:
        ----
            fn: Function to apply element-wise to two tensors.

        Returns:
        -------
            A function to apply `fn` between two tensors.

        """
        f = tensor_zip(fn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Higher-order tensor reduce function.

        Args:
        ----
            fn: Reduction function to apply.
            start: Initial value for reduction.

        Returns:
        -------
            A function to reduce a tensor along a dimension.

        """
        f = tensor_reduce(fn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level tensor map function between tensors with different strides.

    Args:
    ----
        fn: Function to apply element-wise to a tensor.

    Returns:
    -------
        A function that applies `fn` between two tensors.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        in_index = np.zeros(len(in_shape), dtype=np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out[index_to_position(out_index, out_strides)] = fn(
                in_storage[index_to_position(in_index, in_strides)]
            )

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level tensor zip function between tensors with different strides.

    Args:
    ----
        fn: Function to apply element-wise to two tensors.

    Returns:
    -------
        A function that applies `fn` between elements of two tensors.

    """

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
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)
        b_index = np.zeros(len(b_shape), dtype=np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out[index_to_position(out_index, out_strides)] = fn(
                a_storage[index_to_position(a_index, a_strides)],
                b_storage[index_to_position(b_index, b_strides)],
            )

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level tensor reduce function.

    Args:
    ----
        fn: Reduction function to apply element-wise between tensor elements.

    Returns:
    -------
        A function that reduces a tensor along a specific dimension.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            for j in range(a_shape[reduce_dim]):
                a_index = np.array(out_index)
                a_index[reduce_dim] = j
                a_pos = index_to_position(a_index, a_strides)
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
