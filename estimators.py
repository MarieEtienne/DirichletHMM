"""
Estimation initiale de theta = (a0, a1, A0, A1) par maximisation de vraisemblance
du processus latent seul (Dirichlet AR), en deux variantes :

- estimate_raw    : a partir de l'abondance relative RECONSTITUEE X_tilde = Y/N
                     (avec imputation bayesienne des zeros), donc utilisable en pratique.
- estimate_oracle : a partir de l'abondance relative VRAIE X (connue seulement en simulation),
                     sert de borne de reference / comparaison ("que ferait-on si X etait observe ?").

Equivalent Python du bloc R theta_tilde / theta_raw / theta_oracle.
"""

import numpy as np
import nlopt
from scipy.special import gammaln
from pathlib import Path

from simulate import compute_mu, compute_phi


# Repertoire de sortie : sous-dossier data/ a cote de ce script
DATA_DIR = Path(__file__).resolve().parent / "data"


def bobyqa_minimize(obj_f, x0: np.ndarray, xtol_rel: float = 1e-5, maxeval: int = 500) -> np.ndarray:
    """
    Equivalent de nloptr(x0=..., eval_f=obj_f, opts=list(algorithm="NLOPT_LN_BOBYQA",
    xtol_rel=xtol_rel, maxeval=maxeval))$solution.

    obj_f : fonction a MINIMISER, prend un np.ndarray et renvoie un float
            (BOBYQA n'utilise pas de gradient : obj_f n'a pas besoin d'etre differentiable)
    """
    n = len(x0)
    opt = nlopt.opt(nlopt.LN_BOBYQA, n)

    def nlopt_obj(x, grad):
        # grad est toujours vide ici (BOBYQA est derivative-free), donc on l'ignore
        return float(obj_f(np.asarray(x)))

    opt.set_min_objective(nlopt_obj)
    opt.set_xtol_rel(xtol_rel)
    opt.set_maxeval(maxeval)
    return opt.optimize(x0)


# --------------------------------------------------------------------------
# (Re)construction de l'abondance relative observee, avec imputation des zeros
# --------------------------------------------------------------------------

def compute_X_tilde(Y: np.ndarray, N: np.ndarray, p: int) -> np.ndarray:
    """
    X_tilde = Y / N, avec imputation bayesienne (Y+1)/(N+p) sur les colonnes
    contenant au moins un zero (une proportion nulle n'est pas gerable par le
    modele Dirichlet, qui suppose x_i > 0).
    """
    X_tilde = Y / N[np.newaxis, :]
    zero_cols = np.any(X_tilde == 0, axis=0)
    for t in np.where(zero_cols)[0]:
        X_tilde[:, t] = (Y[:, t] + 1) / (N[t] + p)
    return X_tilde


# --------------------------------------------------------------------------
# Critere convexe (initialisation de A0, A1) et log-vraisemblance du processus caché
# --------------------------------------------------------------------------

def estim_convex(A0: np.ndarray, A1: np.ndarray, X: np.ndarray) -> float:
    """
    Pseudo log-vraisemblance convexe, utilisee uniquement pour initialiser A0, A1
    (ignore a0, a1 / la precision). X : (p, Tlen).
    """
    Tlen = X.shape[1]
    total = 0.0
    for t in range(1, Tlen):
        mu_t = compute_mu(A0, A1, X[:, t - 1])
        total += np.sum(np.log(mu_t) * X[:, t])
    return -total / Tlen


def log_lik_hid(a0: float, a1: float, A0: np.ndarray, A1: np.ndarray, X: np.ndarray) -> float:
    """
    Log-vraisemblance du processus latent seul (Dirichlet AR), sans la couche
    d'observation multinomiale. X : (p, Tlen).
    """
    Tlen = X.shape[1]
    total = 0.0
    for t in range(Tlen - 1):
        x_prev, x_curr = X[:, t], X[:, t + 1]
        phi_t = compute_phi(a0, a1, x_prev)
        mu_t = compute_mu(A0, A1, x_prev)
        alpha_t = phi_t * mu_t
        total += gammaln(phi_t) - np.sum(gammaln(alpha_t)) + np.sum((alpha_t - 1) * np.log(x_curr))
    return total


# --------------------------------------------------------------------------
# (Un)packing des vecteurs theta pour l'optimiseur (scipy attend un vecteur plat)
# --------------------------------------------------------------------------

def pack_A(A0: np.ndarray, A1: np.ndarray) -> np.ndarray:
    return np.concatenate([A0.ravel(), A1.ravel()])


def unpack_A(theta: np.ndarray, p: int):
    k = p - 1
    A0 = theta[:k]
    A1 = theta[k:k + k * k].reshape(k, k)
    return A0, A1


def pack_full(a0: float, a1: float, A0: np.ndarray, A1: np.ndarray) -> np.ndarray:
    return np.concatenate([[a0, a1], A0.ravel(), A1.ravel()])


def unpack_full(theta: np.ndarray, p: int):
    k = p - 1
    a0, a1 = theta[0], theta[1]
    A0 = theta[2:2 + k]
    A1 = theta[2 + k:2 + k + k * k].reshape(k, k)
    return a0, a1, A0, A1


# --------------------------------------------------------------------------
# Pipeline d'estimation partage par RAW et ORACLE (seule la donnee d'entree change)
# --------------------------------------------------------------------------

def _estimate_from_trajectory(X: np.ndarray, p: int, estimate_a1: bool = True,
                               xtol_rel: float = 1e-5, maxeval: int = 500):
    """
    X : (p, Tlen) -- soit X_tilde (cas RAW), soit X vrai (cas ORACLE).
    Reproduit : estimateur convexe -> a0_tilde par variance -> MLE complet (nloptr::NLOPT_LN_BOBYQA).

    estimate_a1 : si False, a1 est GELE a 0 (modele a precision constante, cf. scenarios
                  1 et 4) -- non optimise. Le theta retourne garde neanmoins toujours la
                  meme longueur (a1=0.0 dans le vecteur), pour rester homogene avec le
                  reste du pipeline (sgd_em, stockage, boxplots).
    """
    Tlen = X.shape[1]
    k = p - 1

    # ---- 1) estimateur convexe : initialise A0, A1 ----
    A0_init = np.zeros(k)
    A1_init = np.zeros((k, k))
    theta_A_init = pack_A(A0_init, A1_init)

    def obj_convex(theta):
        A0, A1 = unpack_A(theta, p)
        return estim_convex(A0, A1, X)

    theta_A_hat = bobyqa_minimize(obj_convex, theta_A_init, xtol_rel=xtol_rel, maxeval=maxeval)
    A0_hat, A1_hat = unpack_A(theta_A_hat, p)

    # ---- 2) a0_tilde par la variance des residus standardises ----
    mu_hat = np.zeros((p, Tlen - 1))
    for i, t in enumerate(range(1, Tlen)):
        mu_t = compute_mu(A0_hat, A1_hat, X[:, t - 1])
        mu_hat[:, i] = X[:, t] / np.sqrt(mu_t * (1 - mu_t))
    var_per_species = mu_hat.var(axis=1, ddof=1)  # ddof=1 : variance "sample" comme R var()
    a0_init = np.log(1.0 / np.mean(var_per_species))

    # ---- 3) MLE complet du processus latent ----
    if estimate_a1:
        theta_full_init = pack_full(a0_init, 0.0, A0_hat, A1_hat)

        def obj_full(theta):
            a0, a1, A0, A1 = unpack_full(theta, p)
            return -log_lik_hid(a0, a1, A0, A1, X)

        theta_full_hat = bobyqa_minimize(obj_full, theta_full_init, xtol_rel=xtol_rel, maxeval=maxeval)
        return theta_full_hat  # [a0, a1, A0.ravel(), A1.ravel()]
    else:
        # a1 gele a 0 : on optimise sur un vecteur REDUIT (sans a1), puis on
        # reconstruit un theta complet avec a1=0.0 pour garder une longueur homogene.
        theta_reduced_init = np.concatenate([[a0_init], A0_hat.ravel(), A1_hat.ravel()])

        def obj_reduced(theta_reduced):
            a0 = theta_reduced[0]
            A0 = theta_reduced[1:1 + k]
            A1 = theta_reduced[1 + k:1 + k + k * k].reshape(k, k)
            return -log_lik_hid(a0, 0.0, A0, A1, X)

        theta_reduced_hat = bobyqa_minimize(obj_reduced, theta_reduced_init, xtol_rel=xtol_rel, maxeval=maxeval)
        a0_hat = theta_reduced_hat[0]
        A0_hat2 = theta_reduced_hat[1:1 + k]
        A1_hat2 = theta_reduced_hat[1 + k:1 + k + k * k].reshape(k, k)
        return pack_full(a0_hat, 0.0, A0_hat2, A1_hat2)  # a1=0.0, gele


def estimate_raw(Y: np.ndarray, N: np.ndarray, p: int, estimate_a1: bool = True, **kwargs) -> np.ndarray:
    """
    Estimateur RAW : construit X_tilde = Y/N (avec imputation), puis estime theta
    par MLE du processus latent seul. C'est l'estimateur utilisable en pratique
    (X n'est jamais observe dans la vraie vie).
    """
    X_tilde = compute_X_tilde(Y, N, p)
    return _estimate_from_trajectory(X_tilde, p, estimate_a1=estimate_a1, **kwargs)


def estimate_oracle(X: np.ndarray, p: int, estimate_a1: bool = True, **kwargs) -> np.ndarray:
    """
    Estimateur ORACLE : utilise l'abondance relative VRAIE X (connue seulement en
    simulation). Sert de reference pour juger de la perte d'information due au
    bruit d'observation multinomial.
    """
    return _estimate_from_trajectory(X, p, estimate_a1=estimate_a1, **kwargs)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "data.npz"
    data = np.load(data_file)
    X, Y, N = data["X"], data["Y"], data["N"]
    p = int(data["p"])
    a0_true, a1_true = float(data["a0"]), float(data["a1"])
    A0_true, A1_true = data["A0"], data["A1"]

    theta_raw = estimate_raw(Y, N, p)
    theta_oracle = estimate_oracle(X, p)

    print("Vrai theta    : a0=%.3f a1=%.3f A0=%s A1=%s" %
          (a0_true, a1_true, A0_true, A1_true.ravel()))
    print("theta_raw     :", np.round(theta_raw, 3))
    print("theta_oracle  :", np.round(theta_oracle, 3))
