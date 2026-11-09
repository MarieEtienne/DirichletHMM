"""
Etude de simulation : genere N_REP jeux de donnees independants avec les MEMES
parametres vrais, estime theta par les 3 methodes (raw, oracle, sgd), et
sauvegarde tout pour analyse du biais (cf. plot_bias_boxplots.py).

SGD initialise a theta_raw (comparaison equitable, cf. discussion precedente :
partir de theta_true+bruit favorisait artificiellement le SGD).
"""

import numpy as np
from numpy.random import default_rng
import time

from simulate import simulate_dmdhmm
from estimators import estimate_raw, estimate_oracle, pack_full
from sgd_em import sgd_em

# --------------------------------------------------------------------------
# Parametres de l'etude (modifiables)
# --------------------------------------------------------------------------

p = 2
T = 200
a0_true, a1_true = 4.0, -2.0
A0_true = np.array([-1.5])
A1_true = np.array([[2.0]])
theta_true = pack_full(a0_true, a1_true, A0_true, A1_true)

N_REP = 50
N_PART_SGD = 500
N_ITER_SGD = 150
LR_SGD = 0.05

OUTPUT_FILE = "simulation_study_results.npz"

# --------------------------------------------------------------------------
# Boucle principale
# --------------------------------------------------------------------------

if __name__ == "__main__":
    n_theta = len(theta_true)
    results = {
        "raw": np.full((N_REP, n_theta), np.nan),
        "oracle": np.full((N_REP, n_theta), np.nan),
        "sgd": np.full((N_REP, n_theta), np.nan),
    }

    master_rng = default_rng(2024)  # graine "maitre" pour la reproductibilite de l'etude
    t_start = time.time()

    for rep in range(N_REP):
        seed = int(master_rng.integers(0, 2**31 - 1))
        rng = default_rng(seed)

        N = rng.poisson(100, size=T)
        X, Y = simulate_dmdhmm(a0=a0_true, a1=a1_true, A0=A0_true, A1=A1_true,
                                T=T, N=N, p=p, seed=seed)

        theta_raw = None
        try:
            theta_raw = estimate_raw(Y, N, p)
            results["raw"][rep] = theta_raw
        except Exception as e:
            print(f"[rep {rep+1}] estimate_raw a echoue : {e}")

        try:
            theta_oracle = estimate_oracle(X, p)
            results["oracle"][rep] = theta_oracle
        except Exception as e:
            print(f"[rep {rep+1}] estimate_oracle a echoue : {e}")

        if theta_raw is not None:
            try:
                theta_sgd, _ = sgd_em(theta_raw, p, Y, N, n_part=N_PART_SGD,
                                       n_iter=N_ITER_SGD, lr=LR_SGD, rng=rng, verbose=False)
                results["sgd"][rep] = theta_sgd
            except Exception as e:
                print(f"[rep {rep+1}] sgd_em a echoue : {e}")

        elapsed = time.time() - t_start
        print(f"repetition {rep+1}/{N_REP} terminee ({elapsed:.0f}s ecoulees)")

        # ---- sauvegarde incrementale : ecrase le fichier a chaque repetition,
        #      pour ne rien perdre en cas de plantage / interruption ----
        np.savez(
            OUTPUT_FILE,
            theta_true=theta_true,
            raw=results["raw"], oracle=results["oracle"], sgd=results["sgd"],
            p=p, T=T, n_rep=N_REP, n_rep_done=rep + 1,
        )

    print(f"\nResultats sauvegardes dans {OUTPUT_FILE}")