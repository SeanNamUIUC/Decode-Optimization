import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
@triton.jit
# x_ptr: pointer to first vector, y_ptr: pointer to second vector, pointer to output vector, size of the vector, 
#block size: 스레드 블록 하나당 담당처리해야할 데이터 크기, constexpr -> make it constant value to optimize compilation
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, ):
    #몇번째 프로그램인지 -> 몇번째 블록인지.
    #ex) n_elements=256, BLOCK_SIZE=64 -> 4blocks and each block takes 64 data
    pid = tl.program_id(axis=0)
    #내 블록이 시작하는 위치
    block_start = pid * BLOCK_SIZE

    #tl.arange(0,64) -> [0..... 63]
    #내가 맡은 칸 번호 64개 도출
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    #boundary checking
    mask = offsets < n_elements

    #배열의 첫칸 주소를 넣고 주소 64칸을 gpu 메모리에서 한번에 읽어옴 with mask.
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y


    tl.store(output_ptr + offsets, output, mask=mask)


#host code -> calls kernel code (wrapper function)
def add(x: torch.Tensor, y: torch.Tensor):
    #preallocate output -> 커널은 메모리 할당이 안되고 쓰기작업만 가능.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    #
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    #
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output



#똑같은 x 와 y값을 입력받아서 둘의 결과값의 차이를 통해 커널 정확도 확인.
torch.manual_seed(42)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x , y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')



'''Benchmark 그래프와 실험 설정'''
@triton.testing.perf_report( # 
    triton.testing.Benchmark(
        x_names=['size'], #그래프 x축 이름 지정
        x_vals=[2**i for i in range(12,28,1)], #x축으로 테스트할 벡터 크기 목록
        x_log=True, #x축을 로그 스케일로 표시
        line_arg='provider', #그래프에서 각 선line을 구분할 기준 인자 이름(provider)을 지정함
        line_vals=['triton', 'torch'], #그래프 범례(legend)에 표시될 선 이름
        line_names=['triton', 'torch'],#label name for the lines
        styles= [('blue', '-'), ('green', '-')], #각 선의 색상과 모양 (색상, 실선)
        ylabel='GB/s', #그래프 y축 제목 (메모리 처리 대역폭으로 설정)
        plot_name='vector_add_performance', #그래프 제목
        args={}

    ))


#size , provider조합마다 한번씩 호출 --> 실행시간, GB/s 환산
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    #측정된 실행시간 분포에서 중앙값, 하위20, 상위20프로 추출
    quantiles = [0.5,0.2,0.8]
    #gpu 동기화, 웜업을 거쳐서 여러번 반복 실행후 각 분위수에 해당하는 실행시간 반환
    #중앙값, 빠른쪽(작은시간), 느린쪽(큰 시간) 반환
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x+y, quantiles=quantiles)
    if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x,y), quantiles=quantiles)
    #실행시간(ms)을 메모리대역폭(GB/s)로 바꾸는 공식
    #메모리 접근수 (읽기2번 + 쓰기 1번) , 벡터 1개당 바이트수 (원소수 * 4바이트(fp32)) * 1e-9 (10^-9) (Byte -> gigabyte) / (ms * 1e-3(10^-3))
    #메모리접근수 * 원소개수 * 원소의 바이트수 * (Byte -> gigabyte환산) / (ms -> s로 환산 (ms * 1e-3))
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
benchmark.run(print_data=True, show_plots=True)
