import pandas as pd
import matplotlib.pyplot as plt

serial = pd.read_csv(
    "results/runtime_stencil.csv",
    names=["impl", "N", "iters", "runtime"]
)

omp = pd.read_csv(
    "results/runtime_omp_stencil.csv",
    names=["impl", "threads", "N", "iters", "runtime"]
)

cuda = pd.read_csv(
    "results/runtime_cuda_stencil.csv",
    names=["impl", "N", "iters", "runtime"]
)

serial_avg = serial.groupby("N")["runtime"].mean().reset_index()
omp_avg = omp.groupby(["threads", "N"])["runtime"].mean().reset_index()
cuda_avg = cuda.groupby("N")["runtime"].mean().reset_index()

# serial vs OpenMP
plt.figure()
plt.plot(serial_avg["N"], serial_avg["runtime"], marker="o", linestyle="--", label="serial")

for t in sorted(omp_avg["threads"].unique()):
    sub = omp_avg[omp_avg["threads"] == t]
    plt.plot(sub["N"], sub["runtime"], marker="o", label=f"omp {t}")

plt.xlabel("Matrix size N")
plt.ylabel("Runtime (s)")
plt.title("Serial vs OpenMP (Stencil Matrix)")
plt.legend()
plt.tight_layout()
plt.savefig("results/serial_vs_omp_stencil.png", dpi=300)
plt.close()

# CUDA benchmark
plt.figure()
plt.plot(cuda_avg["N"], cuda_avg["runtime"], marker="o", label="cuda")
plt.xlabel("Matrix size N")
plt.ylabel("Runtime (s)")
plt.title("CUDA Benchmark (Stencil Matrix)")
plt.legend()
plt.tight_layout()
plt.savefig("results/cuda_benchmark_stencil.png", dpi=300)
plt.close()

# final comparison
plt.figure()
plt.plot(serial_avg["N"], serial_avg["runtime"], marker="o", linestyle="--", label="serial")

omp8 = omp_avg[omp_avg["threads"] == 8]
plt.plot(omp8["N"], omp8["runtime"], marker="o", label="omp (8 threads)")

plt.plot(cuda_avg["N"], cuda_avg["runtime"], marker="o", label="cuda")

plt.xlabel("Matrix size N")
plt.ylabel("Runtime (s)")
plt.title("Final Comparison (Stencil Matrix)")
plt.legend()
plt.tight_layout()
plt.savefig("results/final_comparison_stencil.png", dpi=300)
plt.close()

print("Stencil benchmark plots saved.")