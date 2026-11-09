"""
Genere une trajectoire simulee du modele DM-DHMM et la sauvegarde sur disque,
pour que l'estimation (estimation.py) puisse la recharger sans re-simuler.
"""

from pathlib import Path
import numpy as np
from numpy.random import default_rng
from simulate import simulate_dmdhmm

# Repertoire de sortie : sous-dossier data/ a cote de ce script
DATA_DIR = Path(__file__).resolve().parent / "data"

# --------------------------------------------------------------------------
# Parametres "vrais" de la simulation (scenario p=2, sans covariable, papier)
# --------------------------------------------------------------------------

SEED = 123
p = 2
T = 200
a0, a1 = 4.0, -2.0
A0 = np.array([-1.5])
A1 = np.array([[2.0]])

if __name__ == "__main__":
    rng = default_rng(SEED)

    N = rng.poisson(100, size=T)
    X, Y = simulate_dmdhmm(a0=a0, a1=a1, A0=A0, A1=A1, T=T, N=N, p=p, seed=SEED)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_DIR / "data.npz"

    np.savez(
        out_file ,
        X=X, Y=Y, N=N,
        a0=a0, a1=a1, A0=A0, A1=A1,
        p=p, T=T, seed=SEED,
    )
    print("Donnees sauvegardees dans data.npz")
    print("X shape:", X.shape, "Y shape:", Y.shape)