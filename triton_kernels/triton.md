# Triton
**vector_addition**
Compute Kernel -> 병렬로 뜬 각 프로그램 pid를 설정후 block_start를 이용해 offsets로 자신들의 담당 범위를 정한뒤 mask로 유효한 원소들만 걸러서 
load -> 연산 -> store로 마무리.

Host wrapper -> GPU메모리에 출력텐서 output을 미리 확보하고 전체 원소의 개수와 BLOCK_SIZE를 기준으로 grid를 설정한후 커널을 띄운뒤에 결과 텐서 output을 돌려주는 함수.

Benchmark function -> Benchmark실행 요약
1.Benchmark x_names, x_vals(x축 변수와 값 목록), 'line_arg' ,'line_vals'(선 구분 변수와 값 목록), args(고정인자) 를 설정 객체로

2.perf_report : 데코레이터가 원본 benchmark함수를 설정과 실행메서드를 가진 래퍼 객체로 감싼다.

3.자동 인자 매핑 -> .run() 조합을 돌며 딕셔너리('args' + 'x_names' + 'line_arg' 값)을 만들어 benchmark로 호출

4.지표 집계 및 출력 -> do_bench -> GB/s 분위수 실행시간 ('ms')를 메모리 트래픽 기준 'GB/s' 로 환산해서 표/그래프/csv 형태로 저장한다.
