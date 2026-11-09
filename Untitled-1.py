"""
Simulation du modele DM-DHMM (Dynamic Multinomial-Dirichlet Hidden Markov Model)
Etienne & Franchi, "Modelling abundance time series derived from count data".

Processus latent :
    X_t | X_{t-1}, Z_t ~ Dirichlet(alpha_t),   alpha_t = phi_t * mu_t

    mu_t (eq. 2)  : transformee logit-multivariee de A0 + A1 x_{t-1}[:-1] + B z_t
    phi_t (eq. 3) : exp(a0 + a1 * Shannon(x_{t-1}))   [precision, controlee par la diversite]

Processus observe :
    Y_t | X_t, N_t ~ Multinomial(N_t, X_t)
"""

import numpy as np
from numpy.random import default_rng


# --------------------------------------------------------------------------
# Fonctions elementaires du modele
# --------------------------------------------------------------------------

def shannon_entropy(x: np.ndarray) -> float:
    """Indice de diversite de Shannon : -sum(x_i log x_i). x doit etre sur le simplexe (x_i > 0)."""
    return float(-np.sum(x * np.log(x)))


def alr_inv(u: np.ndarray) -> np.ndarray:
    """
    Inverse du log-ratio additif (multivariate logit) : R^{p-1} -> simplexe (p coordonnees).
    u : vecteur de taille p-1
    """
    eu = np.exp(u)
    last = 1.0 / (1.0 + eu.sum())
    return np.concatenate([eu * last, [last]])


def compute_mu(A0: np.ndarray, A1: np.ndarray, x_prev: np.ndarray,
                B: np.ndarray | None = None, z: np.ndarray | None = None) -> np.ndarray:
    """
    Moyenne attendue de la distribution de Dirichlet au temps t (eq. 2).

    A0 : vecteur (p-1,)
    A1 : matrice (p-1, p-1)
    x_prev : etat precedent, vecteur (p,) sur le simplexe
    B : matrice (p-1, k) ou None si pas de covariable
    z : vecteur (k,) ou None
    """
    lin = A0 + A1 @ x_prev[:-1]
    if B is not None and z is not None:
        lin = lin + B @ z
    return alr_inv(lin)


def compute_phi(a0: float, a1: float, x_prev: np.ndarray) -> float:
    """Precision de la distribution de Dirichlet au temps t (eq. 3, forme reexprimee avec Shannon)."""
    return float(np.exp(a0 + a1 * shannon_entropy(x_prev)))


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate_dmdhmm(
    a0: float,
    a1: float,
    A0: np.ndarray,
    A1: np.ndarray,
    T: int,
    N: np.ndarray,
    p: int | None = None,
    B: np.ndarray | None = None,
    Z: np.ndarray | None = None,
    alpha0: np.ndarray | None = None,
    seed: int | None = None,
):
    """
    Simule une trajectoire (X_t, Y_t) du modele DM-DHMM.

    Parametres
    ----------
    a0, a1     : parametres scalaires de la precision (eq. 3)
    A0         : vecteur (p-1,), effet de base sur la moyenne
    A1         : matrice (p-1, p-1), interactions biotiques
    T          : longueur de la serie (nombre de pas de temps, t = 0..T-1)
    N          : vecteur (T,) des effectifs totaux observes a chaque temps
    p          : nombre d'especes (deduit de A0 si non fourni)
    B          : matrice (p-1, k) ou None, effet des covariables
    Z          : matrice (T, k) ou None, covariables observees a chaque temps
    alpha0     : parametre de la Dirichlet initiale (defaut : Jeffreys, (1/2,...,1/2))
    seed       : graine du generateur aleatoire

    Retour
    ------
    X : array (p, T)  -- abondances relatives latentes (simplexe)
    Y : array (p, T)  -- comptages observes
    """
    rng = default_rng(seed)

    A0 = np.asarray(A0, dtype=float).reshape(-1)
    A1 = np.asarray(A1, dtype=float).reshape(len(A0), len(A0))
    if p is None:
        p = len(A0) + 1
    N = np.asarray(N)
    assert len(N) == T, "N doit avoir T elements"

    if B is not None:
        B = np.asarray(B, dtype=float).reshape(len(A0), -1)
        assert Z is not None, "Z doit etre fourni si B est fourni"
        Z = np.asarray(Z, dtype=float).reshape(T, -1)

    if alpha0 is None:
        alpha0 = np.full(p, 0.5)  # prior de Jeffreys, non-informatif

    X = np.zeros((p, T))
    Y = np.zeros((p, T))

    # --- t = 0 : initialisation ---
    X[:, 0] = rng.dirichlet(alpha0)
    Y[:, 0] = rng.multinomial(N[0], X[:, 0])

    # --- t = 1 .. T-1 : recursion ---
    for t in range(1, T):
        z_t = Z[t] if Z is not None else None
        mu_t = compute_mu(A0, A1, X[:, t - 1], B=B, z=z_t)
        phi_t = compute_phi(a0, a1, X[:, t - 1])
        alpha_t = phi_t * mu_t

        X[:, t] = rng.dirichlet(alpha_t)
        Y[:, t] = rng.multinomial(N[t], X[:, t])

    return X, Y

# --------------------------------------------------------------------------
# Exemple d'utilisation : reproduit le premier scenario du papier (p=2, sans covariable)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = default_rng(0)

    p = 2
    T = 200
    a0, a1 = 4.0, -2.0
    A0 = np.array([-1.5])   # scalaire ici car p-1 = 1
    A1 = np.array([[2.0]])

    N = rng.poisson(100, size=T)

    X, Y = simulate_dmdhmm(a0=a0, a1=a1, A0=A0, A1=A1, T=T, N=N, p=p, seed=1)

    # Abondance relative "observee" (comptages / effectif), pour verification visuelle
    X_tilde = Y / N

    fig, ax = plt.subplots(figsize=(9, 4))
    for i in range(p):
        ax.plot(X[i], label=f"X_{i+1} (vrai)")
        ax.plot(X_tilde[i], linestyle="--", alpha=0.6, label=f"X_tilde_{i+1} (observe)")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Proportion")
    ax.legend()
    ax.set_title("Simulation DM-DHMM : abondance relative vraie vs observee")
    fig.tight_layout()
    fig.savefig("simulation_example.png", dpi=150)
    print("Figure sauvegardee : simulation_example.png")
    print("X shape:", X.shape, "Y shape:", Y.shape)