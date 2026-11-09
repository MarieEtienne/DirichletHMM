"""
Filtre particulaire SISR (Sequential Importance Sampling with Resampling), eq. (10)-(12)
du papier : proposal q_theta*(x_t | x_{t-1}, y_t) = Dir(alpha_t + y_t).

Fournit :
  - sisr(...)      : trajectoires filtrees + estimation de la vraisemblance marginale (tache 1)
  - compute_Q(...) : Q(theta, theta*) via Monte Carlo sur les trajectoires SISR (tache 2, EM)

AVERTISSEMENT (degenerescence de trajectoire) : cette version reechantillonne les
trajectoires completes a chaque pas de temps (comme le SISR forward standard).
Pour des series longues, les lignees ancestrales degenerent -- Q et son gradient
(fonctionnelles additives sur tout l'horizon) auront une variance qui croit avec T.
C'est le probleme discute precedemment (cf. FFBSm / PaRIS) : cette implementation
pose l'architecture (API stable), le lissage backward pourra etre branche en
remplacement de sisr() sans changer compute_Q / le reste du pipeline.
"""

import numpy as np
from scipy.special import gammaln
from pathlib import Path


from simulate import compute_mu, compute_phi
from joint_log_lik import joint_log_lik


# Repertoire de sortie : sous-dossier data/ a cote de ce script
DATA_DIR = Path(__file__).resolve().parent / "data"


def sisr(a0: float, a1: float, A0: np.ndarray, A1: np.ndarray,
         Y: np.ndarray, N: np.ndarray, n_part: int, rng=None):
    """
    Retourne :
      L_part   : liste de n_part trajectoires, chacune un array (p, Tlen)
      loglik   : estimation de log p(Y_0:T ; theta) (scalaire)
    """
    rng = np.random.default_rng() if rng is None else rng
    p, Tlen = Y.shape

    # ---- t = 0 : init, alpha0 = (1,...,1) comme dans le papier ----
    L_part = [np.empty((p, Tlen)) for _ in range(n_part)]
    for i in range(n_part):
        L_part[i][:, 0] = rng.dirichlet(1.0 + Y[:, 0])

    loglik = 0.0

    for t in range(1, Tlen):
        logweights = np.empty(n_part)
        for i in range(n_part):
            x_prev = L_part[i][:, t - 1]
            phi_t = compute_phi(a0, a1, x_prev)
            mu_t = compute_mu(A0, A1, x_prev)
            alpha_t = phi_t * mu_t

            L_part[i][:, t] = rng.dirichlet(alpha_t + Y[:, t])

            logweights[i] = (
                gammaln(phi_t)
                + np.sum(gammaln(alpha_t + Y[:, t]))
                - gammaln(phi_t + N[t])
                - np.sum(gammaln(alpha_t))
            )

        # ---- estimation de l'increment de vraisemblance (logsumexp stable) ----
        m = np.max(logweights)
        w_unnorm = np.exp(logweights - m)
        loglik += m + np.log(np.mean(w_unnorm))  # log( mean(exp(logweights)) )

        # ---- reechantillonnage (trajectoires completes) ----
        W = w_unnorm / np.sum(w_unnorm)
        idx = rng.choice(n_part, size=n_part, replace=True, p=W)
        L_part = [L_part[i].copy() for i in idx]

    return L_part, loglik


def compute_Q(a0: float, a1: float, A0: np.ndarray, A1: np.ndarray,
              L_part: list, Y: np.ndarray, N: np.ndarray) -> float:
    """
    Q(theta, theta*) = (1/n_part) sum_i joint_log_lik(X_i, Y; theta)

    theta   : parametre EVALUE (celui qu'on optimise dans le M-step)
    L_part  : trajectoires simulees a theta* (sortie de sisr(...) appelee AVEC theta*)
    """
    n_part = len(L_part)
    total = 0.0
    for Xi in L_part:
        total += joint_log_lik(a0, a1, A0, A1, Xi, Y, N)
    return total / n_part


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "data.npz"
    data = np.load(data_file)
    X_true, Y, N = data["X"], data["Y"], data["N"]
    p = int(data["p"])
    a0_true, a1_true = float(data["a0"]), float(data["a1"])
    A0_true, A1_true = data["A0"], data["A1"]

    n_part = 200
    rng = np.random.default_rng()

    # ---- tache 1 : vraisemblance au vrai theta ----
    L_part, loglik = sisr(a0_true, a1_true, A0_true, A1_true, Y, N, n_part=n_part, rng=rng)
    print(f"log-vraisemblance estimee (SISR, n_part={n_part:d}) : {loglik:.3f}")

    # ---- tache 2 : Q(theta, theta*) evalue en theta* = theta_true lui-meme ----
    Q_val = compute_Q(a0_true, a1_true, A0_true, A1_true, L_part, Y, N)
    print(f"Q(theta_true, theta_true) : {Q_val:.3f}")

    # sanity check : Q values a un theta tres different devrait etre plus bas
    Q_bad = compute_Q(0.0, 0.0, np.zeros_like(A0_true), np.zeros_like(A1_true), L_part, Y, N)
    print(f"Q(theta=0, theta_true)    : {Q_bad:.3f}  (doit etre << Q_val)")