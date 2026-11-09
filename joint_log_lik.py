"""
Log-vraisemblance complete (couche latente Dirichlet + couche observee multinomiale),
eq. (6) du papier :

    ell(X_0:T, Y_0:T; theta) =
        sum_{j} (y_{0,j} - 1/2) log(x_{0,j}) - log(y_{0,j}!)         [terme initial]
      + sum_{t=1}^{T} [ log Gamma(phi_t)
                        + sum_j (alpha_{t,j} + y_{t,j} - 1) log(x_{t,j})
                                 - log(y_{t,j}!) - log Gamma(alpha_{t,j}) ]

Deux implementations paralleles :
  - joint_log_lik      : NumPy, rapide, utilisee pour le filtre particulaire et Q(theta,theta')
  - joint_log_lik_jax   : JAX, differentiable, utilisee pour le gradient (Fisher's identity + SGD)

Les deux doivent rester en phase -- si vous modifiez la formule, repercutez le changement
dans les deux fonctions (test de coherence en bas de fichier).
"""

import numpy as np
from scipy.special import gammaln as np_gammaln

import jax.numpy as jnp
from jax.scipy.special import gammaln as jax_gammaln

from simulate import shannon_entropy, alr_inv


# --------------------------------------------------------------------------
# Version NumPy (pour le filtre particulaire : pas besoin de gradient ici)
# --------------------------------------------------------------------------

def joint_log_lik(a0, a1, A0, A1, X, Y, N):
    """
    X, Y : (p, Tlen). N : (Tlen,). A0 : (p-1,). A1 : (p-1, p-1).
    """
    p, Tlen = Y.shape

    total = np.sum((Y[:, 0] - 0.5) * np.log(X[:, 0]) - np_gammaln(Y[:, 0] + 1))

    for t in range(1, Tlen):
        x_prev, x_curr, y_t = X[:, t - 1], X[:, t], Y[:, t]
        phi_t = np.exp(a0 + a1 * shannon_entropy(x_prev))
        lin = A0 + A1 @ x_prev[:-1]
        mu_t = alr_inv(lin)
        alpha_t = phi_t * mu_t

        total += (
            np_gammaln(phi_t)
            + np.sum((alpha_t + y_t - 1) * np.log(x_curr))
            - np.sum(np_gammaln(y_t + 1))
            - np.sum(np_gammaln(alpha_t))
        )
    return total


# --------------------------------------------------------------------------
# Version JAX (differentiable : utilisee uniquement pour le gradient/theta)
# --------------------------------------------------------------------------

def _jax_alr_inv(u):
    eu = jnp.exp(u)
    last = 1.0 / (1.0 + jnp.sum(eu))
    return jnp.concatenate([eu * last, jnp.array([last])])


def _jax_shannon_entropy(x):
    return -jnp.sum(x * jnp.log(x))


def joint_log_lik_jax(theta, p, X, Y, N):
    """
    theta : vecteur JAX plat [a0, a1, A0(p-1,), A1(p-1,p-1)] -- SEUL argument differencie.
    X, Y, N : DONNEES fixes (numpy ou jax arrays, pas de gradient necessaire dessus).

    C'est l'equivalent JAX de la fonction nll/joint_log_lik utilisee dans RTMB :
    theta est ce qu'on differencie, X (les particules) est traite comme une donnee figee.
    """
    k = p - 1
    a0, a1 = theta[0], theta[1]
    A0 = theta[2:2 + k]
    A1 = theta[2 + k:2 + k + k * k].reshape(k, k)

    Tlen = Y.shape[1]
    total = jnp.sum((Y[:, 0] - 0.5) * jnp.log(X[:, 0]) - jax_gammaln(Y[:, 0] + 1))

    def step(carry, t):
        total = carry
        x_prev, x_curr, y_t = X[:, t - 1], X[:, t], Y[:, t]
        phi_t = jnp.exp(a0 + a1 * _jax_shannon_entropy(x_prev))
        lin = A0 + A1 @ x_prev[:-1]
        mu_t = _jax_alr_inv(lin)
        alpha_t = phi_t * mu_t
        contrib = (
            jax_gammaln(phi_t)
            + jnp.sum((alpha_t + y_t - 1) * jnp.log(x_curr))
            - jnp.sum(jax_gammaln(y_t + 1))
            - jnp.sum(jax_gammaln(alpha_t))
        )
        return total + contrib, None

    import jax.lax as lax
    total, _ = lax.scan(step, total, jnp.arange(1, Tlen))
    return total


if __name__ == "__main__":
    # ---- test de coherence NumPy vs JAX sur des donnees jouets ----
    rng = np.random.default_rng(0)
    p, Tlen = 3, 10
    X = rng.dirichlet(np.ones(p), size=Tlen).T
    Y = rng.multinomial(50, [1 / p] * p, size=Tlen).T
    N = np.full(Tlen, 50)

    a0, a1 = 2.0, -0.5
    A0 = np.array([0.1, -0.2])
    A1 = np.array([[0.3, 0.0], [0.1, -0.1]])

    val_np = joint_log_lik(a0, a1, A0, A1, X, Y, N)

    theta = jnp.concatenate([jnp.array([a0, a1]), jnp.array(A0), jnp.array(A1).ravel()])
    val_jax = joint_log_lik_jax(theta, p, jnp.array(X), jnp.array(Y), jnp.array(N))

    print("NumPy :", val_np)
    print("JAX   :", float(val_jax))
    assert np.isclose(val_np, float(val_jax), rtol=1e-5), "Divergence NumPy / JAX !"
    print("OK : les deux implementations concordent.")