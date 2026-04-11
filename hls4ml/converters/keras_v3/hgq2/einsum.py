import typing
from collections.abc import Sequence

from ..einsum_dense import strip_batch_dim
from ._base import QLayerHandler

if typing.TYPE_CHECKING:
    import hgq
    from keras import KerasTensor


class QEinsumHandler(QLayerHandler):
    handles = ('hgq.layers.ops.einsum.QEinsum',)

    def handle(
        self,
        layer: 'hgq.layers.QEinsum',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        assert len(in_tensors) == 2, 'Einsum layer must have exactly two input tensors'
        assert len(out_tensors) == 1, 'Einsum layer must have exactly one output tensor'

        inp0_shape: tuple[int, ...] = in_tensors[0].shape[1:]  # type: ignore
        inp1_shape: tuple[int, ...] = in_tensors[1].shape[1:]  # type: ignore
        out_shape: tuple[int, ...] = out_tensors[0].shape[1:]  # type: ignore
        import pdb;pdb.set_trace()
        if out_shape == ():
            out_shape = (1,)  # Scalar output

        # fmt: off
        assert all(d is not None for d in inp0_shape), \
            f'Error when processing {layer.name}: Einsum layer requires full inp shapes, got {inp0_shape} for inp1'
        assert all(d is not None for d in inp1_shape), \
            f'Error when processing {layer.name}: Einsum layer requires full inp shapes, got {inp1_shape} for inp2'
        assert all(d is not None for d in out_shape), \
            f'Error when processing {layer.name}: EinsumDense layer requires full out shapes. got {out_shape} for output'
        # fmt: on

        equation = strip_batch_dim(layer.equation, einsum_dense=False)
        context_len = layer.context_len
        contract_dim = layer.contract_dim

        if context_len > 1 and contract_dim == 1:
            out_shape = inp1_shape
        
        print(f'OUT_SHAPE: {out_shape}')
        out_tensors[0]._shape = (out_tensors[0].shape[0],) + out_shape
        print(f'{layer.name}: IN0: {inp0_shape}, IN1: {inp1_shape}, OUT: {out_shape}')
        
        return {
            'class_name': 'Einsum',
            'equation': equation,
            'inp0_shape': inp0_shape,
            'inp1_shape': inp1_shape,
            'out_shape': out_shape,
            'context_len': context_len,
            'contract_dim': contract_dim,
        }
