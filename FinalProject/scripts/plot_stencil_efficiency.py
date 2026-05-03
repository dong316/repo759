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

serial_avg = serial.groupby("N")["runtime"].mean().reset_index()
omp_avg = omp.groupby(["threads", "N"])["runtime"].mean().reset_index()

df = pd.merge(
    omp_avg,
    serial_avg,
    on="N",
    suffixes=("_omp", "_serial")
)

df["speedup"] = df["runtime_serial"] / df["runtime_omp"]
df["efficiency"] = df["speedup"] / df["threads"]

# speedup plot
plt.figure()
for t in sorted(df["threads"].unique()):
    sub = df[df["threads"] == t]
    plt.plot(sub["N"], sub["speedup"], marker="o", label=f"omp {t}")

plt.xlabel("Matrix size N")
plt.ylabel("Speedup")
plt.title("OpenMP Speedup (Stencil Matrix)")
plt.legend()
plt.tight_layout()
plt.savefig("results/speedup_stencil.png", dpi=300)
plt.close()

# efficiency plot
plt.figure()
for t in sorted(df["threads"].unique()):
    sub = df[df["threads"] == t]
    plt.plot(sub["N"], sub["efficiency"], marker="o", label=f"omp {t}")

plt.xlabel("Matrix size N")
plt.ylabel("Efficiency")
plt.title("OpenMP Efficiency (Stencil Matrix)")
plt.legend()
plt.tight_layout()
plt.savefig("results/efficiency_stencil.png", dpi=300)
plt.close()

df.to_csv("results/stencil_speedup_efficiency.csv", index=False)

print("Stencil speedup and efficiency plots saved.")