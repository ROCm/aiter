import numpy as np
from itertools import product
from abc import ABC, abstractmethod
from _mlir.ir import VectorType

from flydsl.dialects.ext import flir, arith
from flydsl.dialects.ext.python_control_flow import (
    range_constexpr,
)

fm_fast = flir.arith.FastMathFlags.fast
_asid = flir.const_index
_asv = arith.as_value


class FTensorView:
    def __init__(self, dtype, shape, stride, base_offset, load_impl, store_impl):
        # self.arch = get_rocm_arch()
        self.dtype = dtype
        self.shape = shape
        if stride is None:
            self.stride = tuple(
                (
                    np.cumprod(shape[::-1])[::-1].tolist()
                    + [
                        1,
                    ]
                )[1:]
            )
        else:
            self.stride = stride
        self.base_offset = base_offset  # dynamic
        self.load_impl = load_impl
        self.store_impl = store_impl

    def _linear_offset(self, idxs):
        slice_shape = []
        slice_stride = []
        d_offset = self.base_offset
        for i in range_constexpr(len(idxs)):
            md_id = idxs[i]
            if md_id is None:
                slice_shape.append(self.shape[i])
                slice_stride.append(self.stride[i])
            elif isinstance(md_id, int):
                d_offset = d_offset + md_id * self.stride[i]
            else:
                d_offset = d_offset + md_id * self.stride[i]
        if len(slice_shape) > 0:
            return d_offset, tuple(slice_shape), tuple(slice_stride)
        else:
            return (d_offset,)

    def _lazy_init(self):
        pass

    def __repr__(self):
        return f"TensorView(offset={self.base_offset}, shape={self.shape}, stride={self.stride}, dtype={self.dtype})"

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        if len(offset) == 1:
            return self.load_impl(offset[0])
        else:
            return FTensorView(
                self.dtype,
                offset[1],
                offset[2],
                offset[0],
                self.load_impl,
                self.store_impl,
            )

    def __setitem__(self, idxs, value):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        self.store_impl(offset[0], value)

    def vec_load(self, idxs, vec_size):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        return self.load_impl(offset[0], vec_size=vec_size)

    def vec_store(self, idxs, value, vec_size):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        self.store_impl(offset[0], value, vec_size=vec_size)

    def linear_offset(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        return offset[0]

    def local_tile(self, tile_shape, tile_idxs):
        d_offset = self.base_offset
        stride = []
        for i in range_constexpr(len(tile_idxs)):
            d_offset = d_offset + tile_idxs[i] * tile_shape[i] * self.stride[i]
            stride.append(self.stride[i])
        return FTensorView(
            self.dtype,
            tile_shape,
            tuple(stride),
            d_offset,
            self.load_impl,
            self.store_impl,
        )

    def copy_(self, src_tensor, thread_layout, value_layout, thread_idxs, vec_size):
        src_tensor._lazy_init()
        ndim = len(thread_layout)
        src_offset = src_tensor.base_offset
        dst_offset = self.base_offset
        for d in range_constexpr(ndim):
            src_offset = (
                src_offset + thread_idxs[d] * value_layout[d] * src_tensor.stride[d]
            )
            dst_offset = dst_offset + thread_idxs[d] * value_layout[d] * self.stride[d]
        value_layout_v = value_layout[:-1] + (value_layout[-1] // vec_size,)
        coords = tuple(product(*(range_constexpr(s) for s in value_layout_v)))
        for coord in coords:
            src_vec_offset = src_offset
            dst_vec_offset = dst_offset
            for d in range_constexpr(len(coord)):
                if d == len(coord) - 1:
                    src_vec_offset = (
                        src_vec_offset + coord[d] * src_tensor.stride[d] * vec_size
                    )
                    dst_vec_offset = (
                        dst_vec_offset + coord[d] * self.stride[d] * vec_size
                    )
                else:
                    src_vec_offset = src_vec_offset + coord[d] * src_tensor.stride[d]
                    dst_vec_offset = dst_vec_offset + coord[d] * self.stride[d]
            value = src_tensor.load_impl(src_vec_offset, vec_size=vec_size)
            self.store_impl(dst_vec_offset, value, vec_size=vec_size)


class FTensorBase(ABC):
    def __init__(self, dtype, shape, stride=None, base_offset=0):
        self.tensor_view = None
        self.dtype = dtype
        self.shape = shape
        self.stride = stride
        self.base_offset = base_offset

    @abstractmethod
    def load(self, offset):
        return None

    @abstractmethod
    def store(self, offset, value):
        pass

    def _lazy_init(self):
        if self.tensor_view is None:
            self.tensor_view = FTensorView(
                self.dtype,
                self.shape,
                self.stride,
                self.base_offset,
                self.load,
                self.store,
            )
            self.stride = self.tensor_view.stride
            self.load_impl = self.tensor_view.load_impl
            self.store_impl = self.tensor_view.store_impl

    def __repr__(self):
        self._lazy_init()
        return self.tensor_view.__repr__()

    def __getitem__(self, idxs):
        self._lazy_init()
        return self.tensor_view[idxs]

    def __setitem__(self, idxs, value):
        self._lazy_init()
        self.tensor_view[idxs] = value

    def vec_load(self, idxs, vec_size):
        self._lazy_init()
        return self.tensor_view.vec_load(idxs, vec_size)

    def vec_store(self, idxs, value, vec_size):
        self._lazy_init()
        self.tensor_view.vec_store(idxs, value, vec_size)

    def linear_offset(self, idxs):
        self._lazy_init()
        return self.tensor_view.linear_offset(idxs)

    def local_tile(self, tile_shape, tile_idxs):
        self._lazy_init()
        return self.tensor_view.local_tile(tile_shape, tile_idxs)

    def copy_(self, src_tensor, thread_layout, value_layout, thread_idxs, vec_size):
        self._lazy_init()
        self.tensor_view.copy_(
            src_tensor, thread_layout, value_layout, thread_idxs, vec_size
        )


class TorchTensor(FTensorBase):
    def __init__(self, torch_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.torch_tensor = torch_tensor

    def load(self, offset, vec_size=1):
        return self.torch_tensor.view(-1)[offset : offset + vec_size]

    def store(self, offset, value, vec_size=1):
        self.torch_tensor.view(-1)[offset : offset + vec_size] = value


class GTensor(FTensorBase):
    def __init__(self, fx_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.fx_tensor = fx_tensor
        # self.dtype_bytes = self.dtype.width // 8

    def load(self, offset, vec_size=1):
        if vec_size > 1:
            vec_type = VectorType.get([vec_size], self.dtype)
            return flir.vector.load(
                vec_type, self.fx_tensor, [flir.const_index(offset)], alignment=16
            )
        else:
            return flir.memref.load(self.fx_tensor, [flir.const_index(offset)])

    def store(self, offset, value, vec_size=1):
        if vec_size > 1:
            flir.vector.store(
                arith.as_value(value),
                self.fx_tensor,
                [flir.const_index(offset)],
                alignment=16,
            )
        else:
            flir.memref.store(
                arith.as_value(value),
                self.fx_tensor,
                [
                    flir.const_index(offset),
                ],
            )


class STensor(FTensorBase):
    def __init__(self, fx_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.fx_tensor = fx_tensor.get()

    def load(self, offset, vec_size=1):
        if vec_size > 1:
            vec_type = VectorType.get([vec_size], self.dtype)
            return flir.vector.load(
                vec_type, self.fx_tensor, [flir.const_index(offset)], alignment=16
            )
        else:
            return flir.memref.load(self.fx_tensor, [flir.const_index(offset)])

    def store(self, offset, value, vec_size=1):
        if vec_size > 1:
            flir.vector.store(
                arith.as_value(value),
                self.fx_tensor,
                [flir.const_index(offset)],
                alignment=16,
            )
        else:
            flir.memref.store(
                arith.as_value(value),
                self.fx_tensor,
                [
                    flir.const_index(offset),
                ],
            )
