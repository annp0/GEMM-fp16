import torch
import hgemm
import signal
import sys
import pandas as pd

DEVICE = torch.device("cuda:0")

MAX = 8192
STEP = 256

Ms = [m for m in range(STEP, MAX, STEP)]
Ns = Ms
Ks = Ms

A = torch.randn((MAX, MAX), dtype=torch.half, device=DEVICE)
B = torch.randn((MAX, MAX), dtype=torch.half, device=DEVICE)
C = torch.randn((MAX, MAX), dtype=torch.half, device=DEVICE)

handle_initialized = False

def interrupt_handler(signum, frame):
  print("\nInterrupted, running clean up...")
  if handle_initialized:
    hgemm.destroy_cublas_handle()
    print("cuBLAS handle destroyed")
  sys.exit(1)

signal.signal(signal.SIGINT, interrupt_handler)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

hgemm.init_cublas_handle()
handle_initialized = True

print("Benchmark for fp16 GEMM\n")

print("=== Specs ===")
specs = torch.cuda.get_device_properties(DEVICE)
print("Name:", specs.name)
print("SM Count:", specs.multi_processor_count)
print("Device Memory:", specs.total_memory // (1024 ** 2), "MB\n")

results = []

for M, N, K in zip(Ms, Ns, Ks):
  temp_A = A[:M, :K]
  temp_B = B[:K, :N]
  temp_C = C[:M, :N]

  torch.cuda.synchronize()
  hgemm.hgemm_cublas(temp_A, temp_B, temp_C)
  hgemm.hgemm_cublas(temp_A, temp_B, temp_C)
  torch.cuda.synchronize()
  start_event.record()
  hgemm.hgemm_cublas(temp_A, temp_B, temp_C)
  end_event.record()
  torch.cuda.synchronize()
  ms_cublas = start_event.elapsed_time(end_event)

  torch.cuda.synchronize()
  hgemm.hgemm_bk32_th8x8_async(temp_A, temp_B, temp_C)
  hgemm.hgemm_bk32_th8x8_async(temp_A, temp_B, temp_C)
  torch.cuda.synchronize()
  start_event.record()
  hgemm.hgemm_bk32_th8x8_async(temp_A, temp_B, temp_C)
  end_event.record()
  torch.cuda.synchronize()
  ms_custom = start_event.elapsed_time(end_event)

  results.append({
      "N": M,
      "cublas": ms_cublas,
      "bk32_th8x8_async": ms_custom
  })

df = pd.DataFrame(results)
print(df)

hgemm.destroy_cublas_handle()