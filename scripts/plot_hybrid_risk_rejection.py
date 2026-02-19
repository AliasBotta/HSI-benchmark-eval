"""
plot_hybrid_risk_rejection.py
-----------------------------
Genera le Curve di Risk-Rejection IBRIDE (Clean Style).
Predizione: Clinical Stacking V2.
Incertezza: Epistemic Uncertainty dell'Ensemble.
Palette unificata per coerenza con il Soft Voting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
import sys
import warnings
import gc

warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
metrics_out_dir = FIGURES_DIR / "risk_rejection_hybrid"
metrics_out_dir.mkdir(parents=True, exist_ok=True)

MODELS_DIRS = [
    "rf_20260216_001014",
    "svm-l_20260216_004446",
    "svm-rbf_20260216_015137",
    "dnn_20260216_033845",
    "knn-c_20260216_040940",
    "knn-e_20260216_033918"
]

ID_NT, ID_TT, ID_BV = 0, 1, 2
LABELS_EVAL = [0, 1, 2]

# PALETTE UNIFICATA (Identica all'altro script)
UNIFIED_PALETTE = {
    "Macro F1": "#2ca02c",      # Verde
    "Sens Tumor": "#d62728",    # Rosso
    "Spec Tumor": "#ff7f0e",    # Arancio
    "Sens Vessels": "#1f77b4"   # Blu
}

# =============================================================================
# CLINICAL STACKING V2
# =============================================================================
class ClinicalStackingV2:
    def __init__(self):
        self.vessel_guard = RandomForestClassifier(
            n_estimators=100, max_depth=7, class_weight={1: 10, 0: 1}, n_jobs=-1, random_state=42
        )
        self.onco_resector = RandomForestClassifier(
            n_estimators=100, max_depth=7, class_weight={1: 30, 0: 1}, n_jobs=-1, random_state=42
        )

    def fit(self, X_meta, y_meta):
        y_bv = (y_meta == ID_BV).astype(int)
        self.vessel_guard.fit(X_meta, y_bv)
        mask_not_bv = (y_meta != ID_BV)
        if np.sum(mask_not_bv) > 0:
            X_clean = X_meta[mask_not_bv]
            y_clean = (y_meta[mask_not_bv] == ID_TT).astype(int)
            if len(np.unique(y_clean)) > 1:
                self.onco_resector.fit(X_clean, y_clean)

    def predict(self, X_meta_map):
        H, W, F = X_meta_map.shape
        X_flat = X_meta_map.reshape(-1, F)
        final_flat = np.full(X_flat.shape[0], ID_NT, dtype=np.uint8)

        is_bv = self.vessel_guard.predict(X_flat)
        mask_bv = (is_bv == 1)
        final_flat[mask_bv] = ID_BV

        is_tt = self.onco_resector.predict(X_flat)
        mask_tt = (is_tt == 1) & (~mask_bv)
        final_flat[mask_tt] = ID_TT

        return final_flat.reshape(H, W)

# =============================================================================
# UTILS
# =============================================================================
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
    return u_epi

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

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("🚀 Generazione Curve IBRIDE (Unified Colors)...")

    ref_path = OUTPUTS_DIR / MODELS_DIRS[0]
    folds = sorted([d.name for d in ref_path.glob("fold_*")])
    rejection_rates = np.linspace(0, 0.40, 9)
    all_rows = []

    for test_fold in folds:
        print(f"\n🔹 Processing Fold: {test_fold}")

        # 1. TRAIN STACKER
        train_folds = [f for f in folds if f != test_fold]
        X_train_list = []
        y_train_list = []

        for t_fold in train_folds:
            t_path = ref_path / t_fold
            images = [f.name.replace("_spatial_probs.npy", "") for f in t_path.glob("*_spatial_probs.npy")]
            for img in images:
                gt, mask = load_gt(MODELS_DIRS[0], t_fold, img)
                if gt is None: continue

                valid_indices = np.where(mask.flatten())[0]
                gt_flat = gt.flatten()
                indices_bv = valid_indices[gt_flat[valid_indices] == ID_BV]
                indices_tt = valid_indices[gt_flat[valid_indices] == ID_TT]
                indices_nt = valid_indices[gt_flat[valid_indices] == ID_NT]
                rng = np.random.default_rng(42)
                keep_nt = rng.choice(indices_nt, size=int(len(indices_nt)*0.10), replace=False) if len(indices_nt) > 0 else []
                final_indices = np.concatenate([indices_bv, indices_tt, keep_nt]).astype(int)
                if len(final_indices) == 0: continue

                img_probs = []
                valid_load = True
                for m_dir in MODELS_DIRS:
                    p = load_spatial_probs(m_dir, t_fold, img)
                    if p is None: valid_load = False; break
                    img_probs.append(p.reshape(-1, 4)[final_indices])

                if valid_load:
                    X_train_list.append(np.concatenate(img_probs, axis=1))
                    y_train_list.append(gt_flat[final_indices])

        stacker = ClinicalStackingV2()
        if X_train_list:
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            stacker.fit(X_train, y_train)
            del X_train, y_train
            gc.collect()
        else:
            continue

        # 2. INFERENCE & HYBRID EVALUATION
        test_images = [f.name.replace("_spatial_probs.npy", "") for f in (ref_path / test_fold).glob("*_spatial_probs.npy")]

        for img in test_images:
            gt, mask_valid = load_gt(MODELS_DIRS[0], test_fold, img)
            if gt is None: continue

            probs_list = []
            valid_load = True
            for m_dir in MODELS_DIRS:
                p = load_spatial_probs(m_dir, test_fold, img)
                if p is None: valid_load = False; break
                probs_list.append(p)

            if not valid_load: continue

            # A. PREDIZIONE: Dal Stacker
            stack_input = np.concatenate(probs_list, axis=-1)
            pred_map = stacker.predict(stack_input)

            # B. INCERTEZZA: Epistemica (Dagli input base)
            uncertainty_map = compute_epistemic_uncertainty(probs_list)

            # Filter Valid Pixels
            y_t = gt[mask_valid]
            y_p = pred_map[mask_valid]
            u_e = uncertainty_map[mask_valid]

            if len(y_t) == 0: continue

            # Evaluate Curve
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
                    metrics['fold'] = test_fold
                    metrics['image'] = img
                    all_rows.append(metrics)

    # --- PLOTTING CLEAN ---
    if not all_rows: return

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(metrics_out_dir / "hybrid_risk_rejection.csv", index=False)

    df_melt = df_all.melt(
        id_vars=["rejection_rate"],
        value_vars=["f1_macro", "sens_tt", "spec_tt", "sens_bv"],
        var_name="Metric_Key", value_name="Score"
    )

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

    plt.title("Clinical Stacking Analysis: Risk-Rejection Curve", fontsize=15, fontweight='bold')
    plt.xlabel("Rejection Rate (Excluded % of highest Base-Model Disagreement)", fontsize=12)
    plt.ylabel("Average Score (Image-Level)", fontsize=12)

    plt.ylim(0.5, 1.0)
    plt.xlim(0, 0.4)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))

    # Legenda in basso a destra
    plt.legend(title="Metric", loc="lower right", frameon=True, framealpha=0.95, edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)

    out_path = FIGURES_DIR / "hybrid_risk_rejection_curve_clean.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Grafico Ibrido CLEAN salvato in: {out_path}")

if __name__ == "__main__":
    main()
