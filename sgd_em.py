"""
Tache 3 : preparation du SGD pour maximiser log p(Y;theta) via l'identite de Fisher.

    grad_theta log p(Y;theta) |_{theta=theta*}  =  E[ grad_theta ell(X,Y;theta) |_{theta*} | Y, theta* ]

Pipeline (identique a la version RTMB/R, transpose en JAX) :
  1. sisr(theta*, Y, N, n_part)         -> particules X_i (deja calcule dans particle_filter.py)
  2. pour chaque particule : grad_theta joint_log_lik_jax(theta, X_i, Y, N) |_{theta=theta*}  (JAX autodiff)
  3. moyenne des gradients particulaires -> estimateur stochastique du gradient (Fisher's identity)
  4. pas de descente (Adam) sur theta

Meme avertissement que pour particle_filter.py : les particules viennent d'un SISR
forward simple (pas de lissage backward), donc la variance de ce gradient stochastique
degradera avec T -- c'est le point a ameliorer (FFBSm/PaRIS) si besoin
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path


from particle_filter import sisr
from joint_log_lik import joint_log_lik_jax
from estimators import pack_full, unpack_full, estimate_oracle


# Repertoire de sortie : sous-dossier data/ a cote de ce script
DATA_DIR = Path(__file__).resolve().parent / "data"

# gradient JAX de joint_log_lik_jax par rapport a theta (argnum=0), une seule fois
_grad_joint_log_lik = jax.jit(jax.grad(joint_log_lik_jax, argnums=0), static_argnums=1)


def stochastic_gradient(theta: np.ndarray, p: int, L_part: list, Y: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Estimateur stochastique de grad_theta log p(Y;theta) evalue en theta,
    a partir de particules L_part (obtenues par sisr(...) appele a theta* = theta,
    dans le cas ou vous evaluez au point courant -- Fisher's identity usuelle en SAEM/gradient-EM).

    theta   : vecteur JAX-compatible [a0, a1, A0.ravel(), A1.ravel()]
    L_part  : trajectoires (sortie de sisr)
    """
    theta_jax = jnp.asarray(theta)
    Y_jax, N_jax = jnp.asarray(Y), jnp.asarray(N)

    grads = jnp.stack([
        _grad_joint_log_lik(theta_jax, p, jnp.asarray(Xi), Y_jax, N_jax)
        for Xi in L_part
    ])
    return np.asarray(jnp.mean(grads, axis=0))


def sgd_em(theta_init: np.ndarray, p: int, Y: np.ndarray, N: np.ndarray,
           n_part: int, n_iter: int, lr: float = 1e-2, estimate_a1: bool = True,
           beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
           rng=None, verbose: bool = True):
    """
    Boucle SGD/Adam remplaçant le M-step exact de l'EM (cf. pseudo-code discute) :
    a chaque iteration, un nouveau SISR fournit un gradient stochastique frais
    (pas de moyenne glissante type SAEM ici -- volontairement le cas le plus simple).

    estimate_a1 : si False, a1 est GELE (son gradient est force a 0 a chaque iteration,
                  donc il reste a sa valeur initiale -- typiquement 0.0, cf. scenarios
                  1 et 4 ou la precision est constante, non dependante de la diversite).

    Retourne : theta_final, historique des theta et log-vraisemblances au fil des iterations.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = np.asarray(theta_init, dtype=float).copy()

    m = np.zeros_like(theta)  # premier moment (Adam)
    v = np.zeros_like(theta)  # second moment (Adam)

    history = {"theta": [], "loglik": []}

    for k in range(1, n_iter + 1):
        a0, a1, A0, A1 = unpack_full(theta, p)

        # ---- E-step stochastique : PF au theta courant ----
        L_part, loglik = sisr(a0, a1, A0, A1, Y, N, n_part=n_part, rng=rng)

        # ---- gradient stochastique (Fisher's identity + AD) ----
        grad = stochastic_gradient(theta, p, L_part, Y, N)
        if not estimate_a1:
            grad = grad.copy()
            grad[1] = 0.0  # gele a1 : jamais mis a jour, reste a sa valeur initiale

        # ---- pas Adam (ASCENSION : on maximise log-vraisemblance) ----
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        theta = theta + lr * m_hat / (np.sqrt(v_hat) + eps)

        history["theta"].append(theta.copy())
        history["loglik"].append(loglik)

        if verbose:
            print(f"iter {k:4d} | loglik~{loglik:12.3f} | "
                  f"a0={theta[0]:7.3f} a1={theta[1]:7.3f} "
                  f"|grad|={np.linalg.norm(grad):9.4f}")

    return theta, history


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "data.npz"
    data = np.load(data_file)
    X_true, Y, N = data["X"], data["Y"], data["N"]
    p = int(data["p"])
    a0_true, a1_true = float(data["a0"]), float(data["a1"])
    A0_true, A1_true = data["A0"], data["A1"]

    rng = np.random.default_rng(0)

    # ---- test isole du gradient stochastique au vrai theta ----
    L_part, _ = sisr(a0_true, a1_true, A0_true, A1_true, Y, N, n_part=200, rng=rng)
    theta_true = pack_full(a0_true, a1_true, A0_true, A1_true)
    grad = stochastic_gradient(theta_true, p, L_part, Y, N)
    print("Gradient stochastique en theta_true :", np.round(grad, 3))
    print("(proche de 0 attendu si theta_true est pres d'un optimum local de Q)\n")

    # ---- petite boucle SGD de demonstration, partant d'une valeur perturbee ----
    theta_init = theta_true + rng.normal(scale=0.3, size=theta_true.shape)
    print("theta_init (bruite) :", np.round(theta_init, 3))
    theta_final, history = sgd_em(theta_init, p, Y, N, n_part=100, n_iter=30, lr=0.05, rng=rng)

    # ---- comparaison avec l'estimateur oracle (MLE sur X vrai, cf. estimators.py) ----
    theta_oracle = estimate_oracle(X_true, p)

    print("\ntheta_final  (SGD/Fisher) :", np.round(theta_final, 3))
    print("theta_oracle (MLE sur X)  :", np.round(theta_oracle, 3))
    print("theta_true                :", np.round(theta_true, 3))

    err_sgd = np.linalg.norm(theta_final - theta_true)
    err_oracle = np.linalg.norm(theta_oracle - theta_true)
    print(f"\n||theta_final - theta_true||  = {err_sgd:.4f}")
    print(f"||theta_oracle - theta_true|| = {err_oracle:.4f}")
    print("(l'oracle sert de reference : il connait X, le SGD ne connait que Y)")
