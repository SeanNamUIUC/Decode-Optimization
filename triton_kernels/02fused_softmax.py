import torch

import triton
import triton.language as tl
DEVICE = triton.runtime.driver.active.get_active_torch_device()

#x = (m, n)으로 가정 (배치or 토큰수, 피처차원)
#총 메모리 트래픽: 8MN + 4M
def naive_softmax(x):
    #각 행 안에서 최댓값 찾기
    #DRAM 읽기: MN , DRAM 쓰기: M
    x_max = x.max(dim=1)[0]
    #각 원소에서 해당 행의 최댓값 빼기   (M,) -> (M, 1) 로 broadcasting
    #z[i,j] = x[i,j] - x_max[i]
    #DRAM 읽기: MN + M, DRAM 쓰기: MN
    z = x - x_max[:, None]
    # z의 모든 원소 (0, 1]
    #(M, N)
    #DRAM 읽기: MN , DRAM 쓰기: MN
    numerator = torch.exp(z)

    #(M, )
    ##DRAM 읽기: MN , DRAM 쓰기: M
    denominator = numerator.sum(dim = 1)
    #(M,N) / (M,1)
    #DRAM 읽기: MN+M , DRAM 쓰기: MN
    ret = numerator/denominator[:, None]
    return ret

#X를 단 한번만 읽고 필요한 모든 연산을 on-chip에서 처리하는 custom fused kernel원함.
#이상적으로는 MN 읽고 MN 쓰기 -> 2MN


#Compute Kernel
@triton.jit
#input,output row stride-> 한행 아래로 내려갈때 포인터가 얼마나 이동해야 하는지를 나타내는 보
#n_rows, n_cols, 전체 행과 열 개수
#BLOCK_SIZE: tl.constexpr: 한 프로그램 인스턴스가 한번에 다룰 열 방향 타일 크기기
#주로 block_size는 n_cols보다 크고 이것이 2의 거듭제곱이어야되게 함.
#num_stages: 루프 파이프라이닝 단계 수
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    #몇번째 블록인지 번호 (0 - num_programs-1)
    row_start = tl.program_id(0)
    #전체 블록의 개수(프로그램 개수)
    row_step = tl.num_programs(0)

    #각 프로그램이 전체 행에서 전체 프로그램수를 나눠서 맡을 행들을 가져감. -> 행마다 프로그램 하나
    #num_stages= 루프가 여러번 돌때 이번 행을 계산하는 동안 -> 다음행의 load 미리 시작해
    #메모리 대기시간 숨김.
    #ex) row_start = 0,1,2,3 , n_rows = 10, row_step = 4
    #-> p0->0,4,8    p1 -> 1,5,9 , p2->2,6   p3-> 3,7
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        #1. 읽을 주소 만들기 -> input_ptrs(행 시작주소 + 열 오프셋)

        #행의 시작주소 계산
        row_start_ptr = input_ptr + row_idx * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE) # (0,1...15)
        input_ptrs = row_start_ptr + col_offsets # 주소 16개짜리 벡터

        #앞의 10칸만 진짜 데이터 [T *10 | F*6]
        mask = col_offsets < n_cols



        #2. 읽기 -> tl.load
        #dram에서 한번에 읽어서 레지스터에 올림 -> mask false부분은 읽지 않고
        #-inf로 채움
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))



        #3. 계산해서 softmax_output 만들기
        #[a,b,c, ... 0 | -inf * 6]
        row_minus_max= row - tl.max(row, axis=0)

        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)

        softmax_output = numerator / denominator


        #4. output_ptrs 쓸 주소 만들기
        output_row_start_ptr = output_ptr + row_idx*output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets

        #쓰기
        tl.store(output_ptrs , softmax_output, mask = mask)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    res = torch.empty_like(x)
    #프로그램 하나를 행하나로 잡았기 때문에 grid 모양은 무엇을 프로그램 하나의 단위로 잡았냐 여서 1차원 grid (행수, ) 로 잡는다.
    softmax_kernel[(n_rows, )](res, x, x.stride(0), res.stride(0), n_rows, n_cols,
                               BLOCK_SIZE = BLOCK_SIZE, num_stages=1)
    return res



#Unit testing
def check():
    torch.manual_seed(42)
    x =  torch.randn(1823,781, device=DEVICE)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    #차이를 검사하고 뒷부분에서는 검사 실패시 화면에 보여줄 텐서들
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    # 큰 값 행: max를 빼지 않으면 exp가 inf가 되어 NaN이 나옴
    x = torch.tensor([[1., 2., 3.], [2., 2., 2.], [1000., 1001., 1002.]], device=DEVICE)
    y = softmax(x)
    assert not torch.isnan(y).any()
    assert torch.allclose(y, torch.softmax(x, axis=1)), (y, torch.softmax(x, axis=1))
    print("correct")


#Benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

check()
benchmark.run(show_plots=True, print_data=True)
