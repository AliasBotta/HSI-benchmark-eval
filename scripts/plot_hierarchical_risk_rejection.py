"""
plot_hierarchical_risk_rejection.py
-----------------------------------
Genera le Curve di Risk-Rejection per l'Advanced Hierarchical Ensemble V2.
Logica:
1. Safety Layer (BV): Soglia su probabilità BV (Recall > 98%).
2. Oncological Layer (TT): Soglia ottimizzata F2 o Stacking.
Incertezza: Epistemic Uncertainty dai modelli base.
Stile: Clean & Unified Palette.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve
import sys
import warnings
import gc

warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
metrics_out_dir = FIGURES_DIR / "risk_rejection_hierarchical"
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
THRESHOLD_SAFETY_RECALL_BV = 0.98
BETA_ONCOLOGY = 2.0

# PALETTE UNIFICATA
UNIFIED_PALETTE = {
    "Macro F1": "#2ca02c",      # Verde
    "Sens Tumor": "#d62728",    # Rosso
    "Spec Tumor": "#ff7f0e",    # Arancio
    "Sens Vessels": "#1f77b4"   # Blu
}

# =============================================================================
# HIERARCHICAL OPTIMIZER
# =============================================================================
class HierarchicalOptimizer:
    def __init__(self):
        self.best_tau_bv = 0.1
        self.best_tau_tt = 0.3

    def fit(self, y_true, y_probs):
        # Mappa input 0,1,2 a logica binaria per soglie
        # Attenzione: y_true qui è già shiftato (0=NT, 1=TT, 2=BV)
        y_true_bv = (y_true == ID_BV).astype(int)
        y_score_bv = y_probs[:, ID_BV]
        
        thresholds = np.unique(np.concatenate([np.logspace(-5, -0.5, 100), np.linspace(0.3, 0.9, 20)]))
        thresholds.sort()
        
        for t in thresholds:
            preds = (y_score_bv >= t).astype(int)
            tp = np.sum((preds == 1) & (y_true_bv == 1))
            fn = np.sum((preds == 0) & (y_true_bv == 1))
            recall = tp / (tp + fn + 1e-9)
            if recall < THRESHOLD_SAFETY_RECALL_BV: break
            self.best_tau_bv = t

        mask_passed_safety = (y_probs[:, ID_BV] < self.best_tau_bv)
        if np.sum(mask_passed_safety) > 100:
            y_true_tt = (y_true[mask_passed_safety] == ID_TT).astype(int)
            y_score_tt = y_probs[mask_passed_safety, ID_TT]
            if np.sum(y_true_tt) > 0:
                prec, rec, ths = precision_recall_curve(y_true_tt, y_score_tt)
                with np.errstate(divide='ignore', invalid='ignore'):
                    f2 = (1 + BETA_ONCOLOGY**2) * (prec * rec) / ((BETA_ONCOLOGY**2 * prec) + rec)
                f2 = np.nan_to_num(f2)
                best_idx = np.argmax(f2)
                idx_th = min(best_idx, len(ths)-1)
                self.best_tau_tt = ths[idx_th]

    def predict(self, prob_map):
        H, W, _ = prob_map.shape
        final_map = np.full((H, W), ID_NT, dtype=np.uint8)
        
        # 1. Safety Layer
        mask_bv = prob_map[..., ID_BV] >= self.best_tau_bv
        final_map[mask_bv] = ID_BV
        
        # 2. Oncology Layer
        mask_tt = (prob_map[..., ID_TT] >= self.best_tau_tt) & (~mask_bv)
        final_map[mask_tt] = ID_TT
        
        return final_map

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

def get_model_weights(n_models):
    return np.ones(n_models) / n_models

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
    print("🚀 Generazione Curve Hierarchical Ensemble V2 (Clean Style)...")
    
    ref_path = OUTPUTS_DIR / MODELS_DIRS[0]
    folds = sorted([d.name for d in ref_path.glob("fold_*")])
    rejection_rates = np.linspace(0, 0.40, 9)
    all_rows = []
    
    model_weights = get_model_weights(len(MODELS_DIRS))

    for test_fold in folds:
        print(f"\n🔹 Processing Fold: {test_fold}")
        
        # 1. TRAIN HIERARCHICAL OPTIMIZER (On Train Folds)
        train_folds = [f for f in folds if f != test_fold]
        
        # Accumula predizioni soft su train set per calibrare soglie
        y_probs_train_list = []
        y_true_train_list = []
        
        # print("   ⏳ Calibrating Thresholds on-the-fly...")
        # NOTA: Per ottimizzare, dovremmo caricare tutti i pixel di train.
        # Per semplicità e velocità in questo script di plot, usiamo un subset rappresentativo
        # o carichiamo immagine per immagine per fare fit incrementale (non possibile per curve PR globali).
        # Carichiamo tutto (memory intensive ma corretto)
        
        for t_fold in train_folds:
            t_path = ref_path / t_fold
            images = [f.name.replace("_spatial_probs.npy", "") for f in t_path.glob("*_spatial_probs.npy")]
            for img in images:
                gt, mask = load_gt(MODELS_DIRS[0], t_fold, img)
                if gt is None: continue
                
                # Sampling (Stesso dello script di training per coerenza)
                valid_indices = np.where(mask.flatten())[0]
                gt_flat = gt.flatten()
                
                indices_bv = valid_indices[gt_flat[valid_indices] == ID_BV]
                indices_tt = valid_indices[gt_flat[valid_indices] == ID_TT]
                # Downsample NT/BG per velocità calibrazione
                indices_nt = valid_indices[gt_flat[valid_indices] == ID_NT]
                rng = np.random.default_rng(42)
                keep_nt = rng.choice(indices_nt, size=int(len(indices_nt)*0.05), replace=False) if len(indices_nt) > 0 else []
                
                final_indices = np.concatenate([indices_bv, indices_tt, keep_nt]).astype(int)
                if len(final_indices) == 0: continue
                
                probs_list = []
                valid_load = True
                for m_dir in MODELS_DIRS:
                    p = load_spatial_probs(m_dir, t_fold, img)
                    if p is None: valid_load = False; break
                    probs_list.append(p.reshape(-1, 4)[final_indices])
                
                if valid_load:
                    # Calcola Soft Voting su questi pixel
                    stacked = np.array(probs_list) # (N_models, N_pixels, 4)
                    soft_vote = np.average(stacked, axis=0, weights=model_weights)
                    
                    y_probs_train_list.append(soft_vote)
                    y_true_train_list.append(gt_flat[final_indices])

        # Fit Optimizer
        th_optimizer = HierarchicalOptimizer()
        if y_probs_train_list:
            y_probs_train = np.concatenate(y_probs_train_list, axis=0)
            y_true_train = np.concatenate(y_true_train_list, axis=0)
            th_optimizer.fit(y_true_train, y_probs_train)
            del y_probs_train, y_true_train
            gc.collect()
        else:
            continue

        # 2. INFERENCE & EVALUATION
        # print("   🧪 Evaluating Rejection...")
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
            
            # A. PREDIZIONE: Hierarchical (Soft Voting + Soglie Calibrate)
            stacked_img = np.array(probs_list)
            soft_vote_map = np.average(stacked_img, axis=0, weights=model_weights)
            pred_map = th_optimizer.predict(soft_vote_map) 
            
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
    df_all.to_csv(metrics_out_dir / "hierarchical_risk_rejection.csv", index=False)

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
    
    plt.title("Hierarchical Ensemble Analysis: Risk-Rejection Curve", fontsize=15, fontweight='bold')
    plt.xlabel("Rejection Rate (Excluded % of highest Epistemic Uncertainty)", fontsize=12)
    plt.ylabel("Average Score (Image-Level)", fontsize=12)
    
    plt.ylim(0.5, 1.0)
    plt.xlim(0, 0.4)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))
    
    plt.legend(title="Metric", loc="lower right", frameon=True, framealpha=0.95, edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_path = FIGURES_DIR / "hierarchical_risk_rejection_curve_clean.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Grafico Gerarchico CLEAN salvato in: {out_path}")

if __name__ == "__main__":
    main()
