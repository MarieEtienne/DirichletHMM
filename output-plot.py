"""
Boxplots du biais (estimation - theta_true) par parametre et par methode
(raw / oracle / sgd), a partir de simulation_study_results.npz
(genere par run_simulation_study.py).
"""

import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "simulation_study_results.npz"

# noms des parametres : valable pour p=2 (A0, A1 scalaires) ; a adapter si p>2
PARAM_NAMES = ["a0", "a1", "A0", "A1"]
METHODS = ["raw", "oracle", "sgd"]
COLORS = {"raw": "tab:orange", "oracle": "tab:green", "sgd": "tab:blue"}


def load_and_plot(input_file: str = INPUT_FILE, output_file: str = "bias_boxplots.png"):
    data = np.load(input_file)
    theta_true = data["theta_true"]
    n_params = len(theta_true)
    n_rep = int(data["n_rep"])
    n_rep_done = int(data["n_rep_done"]) if "n_rep_done" in data else n_rep
    if n_rep_done < n_rep:
        print(f"Attention : etude partielle ({n_rep_done}/{n_rep} repetitions terminees)")

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5), sharey=False)
    if n_params == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        biases, labels, colors = [], [], []
        for m in METHODS:
            est = data[m][:, j]
            n_valid = np.sum(~np.isnan(est))
            est = est[~np.isnan(est)]
            biases.append(est - theta_true[j])
            labels.append(f"{m}\n(n={n_valid})")
            colors.append(COLORS[m])

        bp = ax.boxplot(biases, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.axhline(0, color="red", linestyle="--", linewidth=1, label="biais nul")
        ax.set_title(PARAM_NAMES[j] if j < len(PARAM_NAMES) else f"theta[{j}]")
        ax.set_ylabel("biais (estimation - vrai)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Biais des estimateurs sur {n_rep_done}/{n_rep} repetitions "
                 f"(T={int(data['T'])}, p={int(data['p'])})")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    print(f"Figure sauvegardee : {output_file}")

    # ---- resume chiffre, utile en complement visuel ----
    print(f"\n{'parametre':<10} {'methode':<10} {'biais moyen':>12} {'ecart-type':>12}")
    for j in range(n_params):
        pname = PARAM_NAMES[j] if j < len(PARAM_NAMES) else f"theta[{j}]"
        for m in METHODS:
            est = data[m][:, j]
            est = est[~np.isnan(est)]
            bias = est - theta_true[j]
            print(f"{pname:<10} {m:<10} {np.mean(bias):>12.4f} {np.std(bias):>12.4f}")

    return fig


if __name__ == "__main__":
    load_and_plot()