import pandas as pd
import matplotlib.pyplot as plt

# Example plotting script for Jacobi runtime results
# Expected CSV format:
# impl,threads,N,iters,runtime
#
# Example rows:
# serial,1,1024,7077,12.3
# omp,4,1024,7077,3.1

df = pd.read_csv(
    "results/runtime_omp.csv",
    names=["impl", "threads", "N", "iters", "runtime"]
)

plt.figure(figsize=(8, 5))

for threads, sub in df.groupby("threads"):
    sub_mean = sub.groupby("N")["runtime"].mean().reset_index()
    plt.plot(sub_mean["N"], sub_mean["runtime"], marker="o", label=f"omp {threads}")

plt.xlabel("Matrix size N")
plt.ylabel("Runtime (s)")
plt.legend()
plt.tight_layout()
plt.savefig("results/runtime_plot.png", dpi=300)
plt.show()
