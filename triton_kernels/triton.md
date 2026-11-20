# Triton
**기본 틀**
읽을 주소 만들기 -> tl.load -> 레지스터에서 계산 -> 쓸 주소 만들기 -> tl.store
 

**vector_addition**
사고방식: pid가 1D 구간만큼 각 원소끼리 더해서 저장


Compute Kernel -> 병렬로 뜬 각 프로그램 pid를 설정후 block_start를 이용해 offsets로 자신들의 담당 범위를 정한뒤 mask로 유효한 원소들만 걸러서 
load -> 연산 -> store로 마무리.

Host wrapper -> GPU메모리에 출력텐서 output을 미리 확보하고 전체 원소의 개수와 BLOCK_SIZE를 기준으로 grid를 설정한후 커널을 띄운뒤에 결과 텐서 output을 돌려주는 함수.

Benchmark function -> Benchmark실행 요약
1.Benchmark x_names, x_vals(x축 변수와 값 목록), 'line_arg' ,'line_vals'(선 구분 변수와 값 목록), args(고정인자) 를 설정 객체로

2.perf_report : 데코레이터가 원본 benchmark함수를 설정과 실행메서드를 가진 래퍼 객체로 감싼다.

3.자동 인자 매핑 -> .run() 조합을 돌며 딕셔너리('args' + 'x_names' + 'line_arg' 값)을 만들어 benchmark로 호출

4.지표 집계 및 출력 -> do_bench -> GB/s 분위수 실행시간 ('ms')를 메모리 트래픽 기준 'GB/s' 로 환산해서 표/그래프/csv 형태로 저장한다.

**Softmax_kernel**
사고방식: 데이터의 입력값자체는 2차원(M,N)이지만 프로그램이 행( row) 단위로 맡는다. 꼭 1개의 행이아니라 여러행을 나눠 맡을수도 있다.즉 핵심은 행의 길이를 ( n_cols) 기준으로 block_size를 잡아야된다. 주로 n_cols보다 크거나 같은 가장 가까운 2의 거듭제곱으로 잡는다.

2D행렬을 행(row)단위로 쪼개어 프로그램들이 담당(루프를 돌며 행 여러개를 맡을 수도 있음.) BLOCK_SIZE와 행크기( n_cols): 한 행을 한번에 담을수 있도록 n_cols이상의 2의 거듭제곱으로 맞추고 남는 부분은 mask와 -float('inf')로 방어.


Compute Kernel -> 각 프로그램이 program_id(시작 행)와 num_programs(건너뛸 간격)로 맡을 행들을 정하고, 루프마다 row_idx * stride로 행 시작 주소를 잡은 뒤 col_offsets(0 ~ BLOCK_SIZE-1)를 더해 주소 벡터를 만든다. mask로 실제 열(n_cols)만 load하고, 레지스터에서 max 빼기 -> exp -> sum -> 나누기를 한 번에 끝낸 뒤 같은 mask로 store.

Host wrapper -> n_cols로 BLOCK_SIZE(next_power_of_2)를 정하고 출력 텐서를 미리 확보(empty_like)한 뒤, 프로그램 하나 = 행 하나로 grid=(n_rows,)를 잡아 커널을 띄우고 출력 텐서를 return. (커널은 반환값이 없음)
- 텐서를 넘기면 커널 안에서는 첫 원소 포인터가 됨
- BLOCK_SIZE 같은 constexpr는 값이 바뀌면 새로 컴파일됨

Unit test -> 2의 거듭제곱이 아닌 크기(1823 x 781)로 mask를 검증하고, 큰 값 행([1000, 1001, 1002])으로 NaN이 안 나오는지 확인. torch.softmax와 allclose로 비교.

Benchmark -> M=4096 고정, N을 256부터 128씩 늘리며 Triton / torch.softmax / naive를 GB/s로 비교. GB/s = 2 * M * N * 4바이트 / 시간 (이상적인 읽기 1번 + 쓰기 1번 기준)


**핵심 개념**
1. program_id / num_programs / grid
- tl.program_id(0) = 내 프로그램(블록) 번호 = CUDA blockIdx.x
- tl.num_programs(0) = 띄운 프로그램 총개수 = grid 크기 = CUDA gridDim.x
- grid의 차원 수는 데이터 차원이 아니라 "프로그램 하나의 단위를 무엇으로 잡았나"로 정해짐
  - softmax: 행 하나가 단위 -> 1차원 grid (n_rows,)
  - matmul, split-K GEMV: 두 축을 다 나눔 -> 2차원 grid

2. persistent 루프 (grid-stride loop)
- for row_idx in tl.range(program_id, n_rows, num_programs)
- n_rows는 행렬 전체 행 수, 간격은 프로그램 총개수
- 예: 10행, 프로그램 4개 -> p0: 0,4,8 / p1: 1,5,9 / p2: 2,6 / p3: 3,7
- grid=(n_rows,)면 루프가 한 번만 돌아서 행마다 프로그램 하나와 같음

3. stride vs n_cols vs BLOCK_SIZE
- stride(0): 다음 행까지 메모리에서 몇 칸 건너뛰나. 잘라 쓴 텐서(a[:, :3])는 n_cols와 다를 수 있어서 따로 받음
- n_cols: 실제로 담긴 양 (실행 타임 값)
- BLOCK_SIZE: 그릇 크기 (컴파일 타임 상수, 2의 거듭제곱)
- 마지막 행에서 mask가 없으면 텐서 밖을 읽어 illegal memory access

4. mask의 other 값 = 연산의 항등원
- max / softmax -> -inf (max에 영향 없고 exp(-inf)=0이라 sum에도 영향 없음)
- sum (RMSNorm 제곱합) -> 0

5. 수치 안정성
- 행마다 max를 먼저 빼면 가장 큰 값이 e^0 = 1이 돼서 overflow가 없음
- softmax는 모든 원소에 같은 값을 더하거나 빼도 결과가 같아서 가능

6. fusion = DRAM 왕복 줄이기
- naive: 읽기 5MN+2M, 쓰기 3MN+2M -> 총 8MN+4M
- fused: 읽기 MN, 쓰기 MN -> 2MN -> 이론상 약 4배
- 우리 프로젝트도 같은 원리: RMSNorm+residual, dequant+GEMV, W4 양자화(가중치 바이트 1/4)

**다음 커널과 연결**
- RMSNorm: softmax와 같은 행 단위 구조. other=0, tl.sum(h*h)/n_cols -> tl.rsqrt, weight와 residual load 추가
- GEMV: 2D 블록(offs_n[:, None], offs_k[None, :]) + K 방향 루프로 load가 여러 번
