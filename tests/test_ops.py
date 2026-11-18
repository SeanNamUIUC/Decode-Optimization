"""Correctness of every registered impl of every op, on every real decode shape.

    pytest -q                          # everything
    pytest -q -k gemv_w4a16            # one op
    pytest -q -k "gemv_w4a16 and triton_v2"
"""
import pytest
import torch

from specs.model_shapes import LLAMA_3_2_1B, case_label
from specs.op_specs import ALL_SPECS
from tests.conftest import requires_cuda, run_impl_or_skip


def _params():
    for spec in ALL_SPECS.values():
        for impl_name in spec.impls:
            for case in spec.cases(LLAMA_3_2_1B, (1, 3, 8)):  # 3 = non-power-of-2 batch
                yield pytest.param(spec.name, impl_name, case,
                                   id=f"{spec.name}-{impl_name}-{case_label(case).replace(' ', '_')}")


@requires_cuda
@pytest.mark.parametrize("op,impl_name,case", list(_params()))
def test_matches_reference(op, impl_name, case):
    spec = ALL_SPECS[op]
    impl = spec.load_impls()[impl_name]
    args = spec.make_inputs(case)
    expected = spec.ref(*args)
    out = run_impl_or_skip(impl, args)
    torch.testing.assert_close(out, expected, atol=spec.atol, rtol=spec.rtol)
