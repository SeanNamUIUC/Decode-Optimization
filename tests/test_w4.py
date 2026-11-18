"""Day 6 -- the W4A16 format itself. Runs on CPU too."""
import torch

from quant.w4 import dequantize_w4, pack_int4, quantize_w4, unpack_int4


def test_pack_unpack_roundtrip():
    q = torch.randint(0, 16, (64, 256))
    assert torch.equal(unpack_int4(pack_int4(q)), q)


def test_pack_layout_matches_doc():
    # q[n, 8*j + i] lives in bits [4i, 4i+4) of qweight[n, j]
    q = torch.zeros(1, 8, dtype=torch.int64)
    q[0, 3] = 0xA
    assert pack_int4(q)[0, 0].item() == 0xA << 12


def test_high_nibble_sign_is_safe():
    q = torch.full((2, 8), 15)  # all bits set -> int32 -1
    packed = pack_int4(q)
    assert packed.dtype == torch.int32 and packed[0, 0].item() == -1
    assert torch.equal(unpack_int4(packed), q)


def test_quant_error_is_bounded():
    torch.manual_seed(0)
    w = torch.randn(256, 1024) / 32
    qw, s = quantize_w4(w, 128)
    w_hat = dequantize_w4(qw, s, 128).float()
    # per-element error <= scale / 2 (plus fp16 rounding)
    max_scale = s.float().repeat_interleave(128, dim=1)
    assert ((w - w_hat).abs() <= max_scale / 2 + 1e-3).all()
    rel = (w - w_hat).norm() / w.norm()
    print(f"relative quantization error: {rel:.4f}")
    assert rel < 0.15
