import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def run_impl_or_skip(impl, args):
    """Run an implementation; skip the test if the kernel isn't written yet."""
    try:
        run_args = impl.prepare(args) if impl.prepare else args
        return impl.fn(*run_args)
    except NotImplementedError as e:
        pytest.skip(str(e))
