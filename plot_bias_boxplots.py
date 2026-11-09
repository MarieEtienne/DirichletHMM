"""
Boxplots du biais (estimation - theta_true) par parametre et par methode
(raw / oracle / sgd), a partir d'un fichier study_<scenario>.npz
(genere par run_scenario_study.py).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

METHODS = ["raw", "oracle", "sgd"]
COLORS = {"raw": "tab:orange", "oracle": "tab:green", "sgd": "tab:blue"}


def param_names(p: int, estimate_a1: bool) -> list:
    """Genere les noms de parametres dans l'ordre du vecteur theta = [a0, a1, A0.ravel(), A1.ravel()]."""
    k = p - 1
    names = ["a0", "a1"]  # a1 reste dans le vecteur meme si gele (valeur fixee a 0)
    names += [f"A0[{i+1}]" for i in range(k)]
    names += [f"A1[{i+1},{j+1}]" for i in range(k) for j in range(k)]
    return names


def load_and_plot(scenario_name: str):
    input_file = f"study_{scenario_name}.npz"
    output_file = f"bias_boxplots_{scenario_name}.png"

    data = np.load(input_file)
    theta_true = data["theta_true"]
    p = int(data["p"])
    estimate_a1 = bool(data["estimate_a1"]) if "estimate_a1" in data else True
    n_rep = int(data["n_rep"])
    n_rep_done = int(data["n_rep_done"]) if "n_rep_done" in data else n_rep
    if n_rep_done < n_rep:
        print(f"Attention : etude partielle ({n_rep_done}/{n_rep} repetitions terminees)")

    names = param_names(p, estimate_a1)
    # on n'affiche PAS le panneau a1 s'il est gele (bias trivialement nul, pas informatif)
    plot_indices = [j for j in range(len(theta_true)) if estimate_a1 or names[j] != "a1"]
    n_panels = len(plot_indices)

    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    print(f"\n{'parametre':<10} {'methode':<10} {'biais moyen':>12} {'ecart-type':>12}")
    for ax, j in zip(axes, plot_indices):
        biases, labels, colors = [], [], []
        for m in METHODS:
            est = data[m][:, j]
            n_valid = np.sum(~np.isnan(est))
            est = est[~np.isnan(est)]
            bias = est - theta_true[j]
            biases.append(bias)
            labels.append(f"{m}\n(n={n_valid})")
            colors.append(COLORS[m])
            if n_valid > 0:
                print(f"{names[j]:<10} {m:<10} {np.mean(bias):>12.4f} {np.std(bias):>12.4f}")

        bp = ax.boxplot(biases, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(names[j])
        ax.set_ylabel("biais (estimation - vrai)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-4,4)

    fig.suptitle(f"{scenario_name} -- biais sur {n_rep_done}/{n_rep} repetitions "
                 f"(T={int(data['T'])}, p={p}, a1 {'estime' if estimate_a1 else 'gele a 0'})")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    print(f"\nFigure sauvegardee : {output_file}")

    return fig


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario_names = sys.argv[1:]
    else:
        from scenarios import SCENARIOS
        scenario_names = [s["name"] for s in SCENARIOS]

    for name in scenario_names:
        try:
            load_and_plot(name)
        except FileNotFoundError:
            print(f"Pas de resultats pour '{name}' (study_{name}.npz introuvable) -- ignore.")
