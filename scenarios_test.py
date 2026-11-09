"""
Definition des scenarios de simulation a comparer.

Chaque scenario est un dict avec :
  name        : identifiant (utilise pour nommer les fichiers de resultats)
  p           : nombre d'especes
  a0          : parametre de precision (toujours estime)
  a1          : parametre de precision (dependance a la diversite) ; None => GELE a 0,
                non estime (modele a precision constante)
  A0          : vecteur (p-1,)
  A1          : matrice (p-1, p-1)
  alpha0      : parametre Dirichlet pour X_0 a la simulation (None => defaut Jeffreys
                (1/2,...,1/2) de simulate_dmdhmm)
  T           : longueur de la serie simulee
"""

import numpy as np

SCENARIOS_TEST = [
    dict(
        name="scenario1_a1_fixed_p2",
        p=2,
        a0=5.0,
        a1=None,                        # gele a 0 (theta_true R : c(a0, A0, A1), pas de a1)
        A0=np.array([1.0]),
        A1=np.array([[-2.0]]),
        alpha0=None,
        T=200,
    ),
    dict(
        name="scenario2_full_p2",
        p=2,
        a0=5.0,
        a1=-3.0,
        A0=np.array([1.0]),
        A1=np.array([[-2.0]]),
        alpha0=None,
        T=200,
    ),
    dict(
        name="scenario3_full_p2_A0_4",
        p=2,
        a0=5.0,
        a1=-3.0,
        A0=np.array([4.0]),
        A1=np.array([[-2.0]]),
        alpha0=None,
        T=200,
    ),
    dict(
        name="scenario4_a1_fixed_p4",
        p=4,
        a0=5.0,
        a1=None,                        # gele a 0 (theta_true R : c(a0, A0, t(A1)), pas de a1)
        A0=np.array([-0.5, -1.0, -0.5]),
        A1=np.array([
            [1.0, -1.0, 1.0],
            [1.0, 3.0, 0.0],
            [-1.0, 2.0, 1.0],
        ]),
        alpha0=np.ones(4),               # explicitement rep(1,p) dans le code R
        T=200,
    ),
    dict(
        name="scenario5",
        p=2,
        a0=5.0,
        a1=-5.0,
        A0=np.array([0.5]),
        A1=np.array([[-3.0]]),
        alpha0=None,
        T=200,
    ),
]


def get_scenario(name: str) -> dict:
    for s in SCENARIOS:
        if s["name"] == name:
            return s
    raise KeyError(f"Scenario inconnu : {name}. Disponibles : {[s['name'] for s in SCENARIOS]}")
