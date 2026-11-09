"""
Etude de simulation multi-scenarios : pour chaque scenario de scenarios.py, genere
N_REP jeux de donnees independants, estime theta par les 3 methodes (raw, oracle, sgd),
et sauvegarde (incrementalement) dans un fichier study_<scenario_name>.npz separe.

Les donnees simulees elles-memes (X, Y, N) sont aussi sauvegardees par repetition,
dans un fichier a part par scenario (datasets_<scenario_name>.npz), pour pouvoir
les reutiliser plus tard (ex. relancer une autre methode d'estimation sans re-simuler).
"""

import numpy as np
from numpy.random import default_rng
import time
from pathlib import PATH

from simulate import simulate_dmdhmm
from estimators import estimate_raw, estimate_oracle, pack_full
from sgd_em import sgd_em
from scenarios import SCENARIOS

N_REP = 3
N_PART_SGD = 100
N_ITER_SGD = 150
LR_SGD = 0.05
MASTER_SEED = 2024


def run_scenario(scenario: dict, n_rep: int = N_REP,
                  n_part_sgd: int = N_PART_SGD, n_iter_sgd: int = N_ITER_SGD, lr_sgd: float = LR_SGD):
    name = scenario["name"]
    p, T = scenario["p"], scenario["T"]
    lamb = scenario["lambda"]
    a0, a1, A0, A1, alpha0 = scenario["a0"], scenario["a1"], scenario["A0"], scenario["A1"], scenario["alpha0"]
    estimate_a1 = a1 is not None
    a1_sim = a1 if a1 is not None else 0.0  # valeur utilisee pour SIMULER (0 si gele)

    theta_true = pack_full(a0, a1_sim, A0, A1)
    n_theta = len(theta_true)

    results = {
        "raw": np.full((n_rep, n_theta), np.nan),
        "oracle": np.full((n_rep, n_theta), np.nan),
        "sgd": np.full((n_rep, n_theta), np.nan),
    }
    # stockage des donnees simulees, pour reutilisation ulterieure
    X_all = np.full((n_rep, p, T), np.nan)
    Y_all = np.full((n_rep, p, T), np.nan)
    N_all = np.full((n_rep, T), np.nan)
    seeds_all = np.full(n_rep, -1, dtype=np.int64)

    master_rng = default_rng(MASTER_SEED)
    t_start = time.time()

    print(f"\n=== Scenario : {name} (p={p}, T={T}, estimate_a1={estimate_a1}) ===")

    for rep in range(n_rep):
        seed = int(master_rng.integers(0, 2**31 - 1))
        rng = default_rng(seed)
        seeds_all[rep] = seed

        N = rng.poisson(lamb, size=T)
        X, Y = simulate_dmdhmm(a0=a0, a1=a1_sim, A0=A0, A1=A1, T=T, N=N, p=p,
                                alpha0=alpha0, seed=seed)
        X_all[rep], Y_all[rep], N_all[rep] = X, Y, N

        theta_raw = None
        try:
            theta_raw = estimate_raw(Y, N, p, estimate_a1=estimate_a1)
            results["raw"][rep] = theta_raw
        except Exception as e:
            print(f"[{name} | rep {rep+1}] estimate_raw a echoue : {e}")

        try:
            theta_oracle = estimate_oracle(X, p, estimate_a1=estimate_a1)
            results["oracle"][rep] = theta_oracle
        except Exception as e:
            print(f"[{name} | rep {rep+1}] estimate_oracle a echoue : {e}")

        if theta_raw is not None:
            try:
                theta_sgd, _ = sgd_em(theta_raw, p, Y, N, n_part=n_part_sgd, n_iter=n_iter_sgd,
                                       lr=lr_sgd, estimate_a1=estimate_a1, rng=rng, verbose=False)
                results["sgd"][rep] = theta_sgd
            except Exception as e:
                print(f"[{name} | rep {rep+1}] sgd_em a echoue : {e}")

        elapsed = time.time() - t_start
        print(f"[{name}] repetition {rep+1}/{n_rep} terminee ({elapsed:.0f}s ecoulees)")

        # ---- sauvegarde incrementale (resultats ET donnees) ----
        
        np.savez(
            f"study_{name}.npz",
            theta_true=theta_true, estimate_a1=estimate_a1,
            raw=results["raw"], oracle=results["oracle"], sgd=results["sgd"],
            p=p, T=T, n_rep=n_rep, n_rep_done=rep + 1,
        )
        np.savez(
            f"datasets_{name}.npz",
            X=X_all, Y=Y_all, N=N_all, seeds=seeds_all,
            p=p, T=T, n_rep=n_rep, n_rep_done=rep + 1,
        )

    print(f"[{name}] termine -> study_{name}.npz, datasets_{name}.npz")


if __name__ == "__main__":
    for scenario in SCENARIOS:
        run_scenario(scenario)
