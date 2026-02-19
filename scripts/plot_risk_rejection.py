"""
plot_risk_rejection.py
----------------------
Genera le Curve di Risk-Rejection per il SOFT VOTING (Clean Style).
Palette unificata per coerenza con l'Hybrid Stacking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import warnings

# Ignora warning
warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
metrics_out_dir = FIGURES_DIR / "risk_rejection"
metrics_out_dir.mkdir(parents=True, exist_ok=True)

MODELS_DIRS = [
    "rf_20260216_001014",
    "svm-l_20260216_004446",
    "svm-rbf_20260216_015137",
    "dnn_20260216_033845",
    "knn-c_20260216_040940",
    "knn-e_20260216_033918"
]

LABELS_EVAL = [0, 1, 2]

# PALETTE UNIFICATA
UNIFIED_PALETTE = {
    "Macro F1": "#2ca02c",      # Verde
    "Sens Tumor": "#d62728",    # Rosso
    "Spec Tumor": "#ff7f0e",    # Arancio
    "Sens Vessels": "#1f77b4"   # Blu
}

def load_spatial_probs(model_dir, fold, image):
    path = OUTPUTS_DIR / model_dir / fold / f"{image}_spatial_probs.npy"
    return np.load(path) if path.exists() else None

def load_gt(ref_model_dir, fold, image):
    path = OUTPUTS_DIR / ref_model_dir / fold / f"{image}_gt.npy"
    if not path.exists(): return None, None
    gt = np.load(path)
    mask_valid = (gt != 0) & (gt != 4)
    gt_shifted = np.zeros_like(gt)
    gt_shifted[mask_valid] = gt[mask_valid] - 1
    return gt_shifted, mask_valid

def fast_entropy(probs, base=4):
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1) / np.log(base)

def compute_epistemic_uncertainty(probs_list):
    stacked = np.array(probs_list)
    p_mean = np.mean(stacked, axis=0)
    u_total = fast_entropy(p_mean)
    entropies = fast_entropy(stacked)
    u_aleatoric = np.mean(entropies, axis=0)
    u_epi = u_total - u_aleatoric
    return u_epi, p_mean

def calculate_metrics_single_image(y_true, y_pred):
    if len(y_true) == 0: return None
    f1 = f1_score(y_true, y_pred, average='macro', labels=LABELS_EVAL)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_EVAL)

    def bin_metrics(idx):
        if np.sum(cm[idx, :]) == 0: return 0.0, 1.0
        tp = cm[idx, idx]
        fn = np.sum(cm[idx, :]) - tp
        fp = np.sum(cm[:, idx]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        return sens, spec

    sens_tt, spec_tt = bin_metrics(1)
    sens_bv, _       = bin_metrics(2)

    return {
        "f1_macro": f1,
        "sens_tt": sens_tt,
        "spec_tt": spec_tt,
        "sens_bv": sens_bv
    }

def main():
    print("🚀 Generazione Risk-Rejection SOFT VOTING (Unified Colors)...")

    ref_path = OUTPUTS_DIR / MODELS_DIRS[0]
    folds = sorted([d.name for d in ref_path.glob("fold_*")])
    rejection_rates = np.linspace(0, 0.40, 9)

    all_rows = []

    for fold in folds:
        print(f"🔹 Processing {fold}...")
        images = [f.name.replace("_spatial_probs.npy", "") for f in (ref_path / fold).glob("*_spatial_probs.npy")]

        for img in images:
            gt, mask_valid = load_gt(MODELS_DIRS[0], fold, img)
            if gt is None: continue

            probs_list = []
            valid_load = True
            for m_dir in MODELS_DIRS:
                p = load_spatial_probs(m_dir, fold, img)
                if p is None: valid_load = False; break
                probs_list.append(p)
            if not valid_load: continue

            u_epi_map, p_mean_map = compute_epistemic_uncertainty(probs_list)
            pred_map = np.argmax(p_mean_map, axis=-1)

            y_t = gt[mask_valid]
            y_p = pred_map[mask_valid]
            u_e = u_epi_map[mask_valid]

            if len(y_t) == 0: continue

            for rate in rejection_rates:
                if rate == 0:
                    mask_keep = np.ones_like(u_e, dtype=bool)
                else:
                    cutoff = np.percentile(u_e, 100 * (1 - rate))
                    mask_keep = u_e <= cutoff

                y_t_filt = y_t[mask_keep]
                y_p_filt = y_p[mask_keep]

                metrics = calculate_metrics_single_image(y_t_filt, y_p_filt)
                if metrics:
                    metrics['rejection_rate'] = rate
                    metrics['fold'] = fold
                    metrics['image'] = img
                    all_rows.append(metrics)

    if not all_rows: return

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(metrics_out_dir / "risk_rejection_image_level.csv", index=False)

    df_melt = df_all.melt(
        id_vars=["rejection_rate"],
        value_vars=["f1_macro", "sens_tt", "spec_tt", "sens_bv"],
        var_name="Metric_Key", value_name="Score"
    )

    # Mapping per visualizzazione e colori
    key_to_label = {
        "f1_macro": "Macro F1",
        "sens_tt": "Sens Tumor",
        "spec_tt": "Spec Tumor",
        "sens_bv": "Sens Vessels"
    }
    df_melt["Metric"] = df_melt["Metric_Key"].map(key_to_label)

    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    sns.lineplot(
        data=df_melt, x="rejection_rate", y="Score", hue="Metric", style="Metric",
        markers=True, dashes=False, palette=UNIFIED_PALETTE,
        linewidth=2.5, markersize=8,
        err_style="bars", errorbar=("ci", 95), err_kws={'capsize': 4, 'elinewidth': 1.5}
    )

    plt.title("Soft Voting Analysis: Risk-Rejection Curve", fontsize=15, fontweight='bold')
    plt.xlabel("Rejection Rate (Excluded % of most uncertain pixels)", fontsize=12)
    plt.ylabel("Average Score (Image-Level)", fontsize=12)

    plt.ylim(0.5, 1.0)
    plt.xlim(0, 0.4)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))

    plt.legend(title="Metric", loc="lower right", frameon=True, framealpha=0.95, edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)

    out_path = FIGURES_DIR / "risk_rejection_curve_corrected.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Grafico Soft Voting (Unified Colors) salvato in: {out_path}")

if __name__ == "__main__":
    main()
