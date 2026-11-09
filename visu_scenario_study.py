"""
visualiser un cas particulier.
"""

import numpy as np
from numpy.random import default_rng
import time
import matplotlib.pyplot as plt

from simulate import simulate_dmdhmm
from estimators import estimate_raw, estimate_oracle, pack_full
from scenarios import SCENARIOS

N_REP = 3
N_PART_SGD = 100
N_ITER_SGD = 150
LR_SGD = 0.05
MASTER_SEED = 2024


def visualize_scenario(scenario: dict, n_rep: int = N_REP):
    name = scenario["name"]
    p, T = scenario["p"], scenario["T"]
    a0, a1, A0, A1, alpha0 = scenario["a0"], scenario["a1"], scenario["A0"], scenario["A1"], scenario["alpha0"]
    a1_sim = a1 if a1 is not None else 0.0  # valeur utilisee pour SIMULER (0 si gele)
    theta_true = pack_full(a0, a1_sim, A0, A1)
    n_theta = len(theta_true)

    
    output_file = f"{name}_simu.png"
    output_file2 = f"{name}_entropie.png"

    master_rng = default_rng(MASTER_SEED)
    t_start = time.time()

    print(f"\n=== Scenario : {name} (p={p}, T={T}")

    seed = int(master_rng.integers(0, 2**31 - 1))
    rng = default_rng(seed)
    
    N = rng.poisson(100, size=T)
    X, Y = simulate_dmdhmm(a0=a0, a1=a1_sim, A0=A0, A1=A1, T=T, N=N, p=p,
                                alpha0=alpha0, seed=seed)

    x = np.arange(T)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(p):
        ax.plot(x, Y[i, :], label=f"cat. {i}")   # use Y[:, i] if Y is (T, p)
    ax.set_xlabel("t")
    ax.set_ylabel("counts")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
 
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    entropie = np.zeros(T)
    for t in range(T):
        prov = 0
        for i in range(p):
            prov = X[i,t]* np.log(X[i,t]) + prov
        entropie[t] = prov
    ax2.plot(x, entropie, label=f"cat. {i}")   

    ax2.set_xlabel("t")
    ax2.set_ylabel("entropie")
    ax2.set_title(name)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_file2, dpi=150)
    plt.close(fig2)

if __name__ == "__main__":
    for scenario in SCENARIOS:
        visualize_scenario(scenario)
