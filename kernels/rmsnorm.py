"""Day 3 -- fused RMSNorm + residual add.  (YOU write this one.)

Contract (must match reference.ops.rmsnorm_residual exactly):
    rmsnorm_residual(x, residual, weight, eps) -> (out, h)
    x, residual: [M, N] fp16 contiguous (residual may be None)
    weight:      [N] fp16
    h   = x + residual            (new residual stream, fp16)
    out = h * rsqrt(mean(h^2) + eps) * weight   (statistics in fp32)

Goal: one read of x/residual/weight, one write of out/h.
Hint: one program per row; N=2048 fits in one BLOCK (next_power_of_2).
"""
import torch
import triton
import triton.language as tl


@triton.jit
#x_ptr: 입력 메모리의 시작점
def _rmsnorm_residual_kernel(x_ptr, res_ptr, w_ptr, out_ptr, h_ptr,
                             N, stride, eps,
                             HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) #block(프로그램) 번호 0,1,2,3
    #내 행안의 칸들 위치 잡기
    cols = tl.arange(0, BLOCK) 
    mask = cols < N

    offset = row * stride + cols
    # 1. 읽기
    #mask가 false 인 부분은 0으로 만들게 해서 제곱합 0으로 소용없게함.
    x = tl.load(x_ptr + offset, mask=mask, other=0.)
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + offset, mask = mask, other=0,)#residual 읽기 (x와 같은 방식)
        h = x + r
        tl.store(h_ptr + offset, h, mask = mask)#h를 h_ptr에 store (다음 layer의 residual로 쓰임)
    else:
        h = x

    # 2. 계산 (통계는 fp32로)
    hf = h.to(tl.float32)
    hf_square = hf * hf
    var = tl.sum(hf_square, axis=-1) / N# 행의 제곱평균. tl.sum 사용. N으로 나눌 것 (BLOCK이 아니라!)
    rstd = tl.rsqrt(var + eps) #tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask = mask, other=0,)# weight 읽기 (weight는 [N] 1차원이라 row 오프셋 없음)
    out =(hf*rstd).to(tl.float16) * w#(hf * rstd)를 fp16으로 바꾼 뒤 w 곱하기

    # 3. 쓰기
    # out을 out_ptr에 store
    tl.store(out_ptr+offset, out, mask=mask)


def rmsnorm_residual(x, residual, weight, eps=1e-5):
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    has_res = residual is not None
    h = torch.empty_like(x) if has_res else x       # residual 없으면 h = x 그대로 반환 (store 불필요)
    res = residual if has_res else x                 # 안 쓰는 자리엔 아무 텐서나 (HAS_RESIDUAL=False라 안 읽음)
    BLOCK = triton.next_power_of_2(N)
    _rmsnorm_residual_kernel[(M,)](x, res, weight, out, h, N, x.stride(0), eps,
                                   HAS_RESIDUAL=has_res, BLOCK=BLOCK, num_warps=8)
    return out, h