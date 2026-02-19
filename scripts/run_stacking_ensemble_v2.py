import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import sys
import gc
from datetime import datetime

# --- CONFIGURAZIONE ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.metrics import compute_all_metrics
from utils.helpers import ensure_dir

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# USA LE CARTELLE CORRETTE QUI
MODELS_DIRS = [
    "rf_20260216_001014",
    "svm-l_20260216_004446",
    "svm-rbf_20260216_015137",
    "dnn_20260216_033845",
    "knn-c_20260216_040940",
    "knn-e_20260216_033918"
]

ID_NT, ID_TT, ID_BV, ID_BG = 0, 1, 2, 3
THRESHOLD_SAFETY_RECALL_BV = 0.98
BETA_ONCOLOGY = 2.0

ENSEMBLE_DIR_NAME = f"stacking_v2_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
ENSEMBLE_PATH = OUTPUT_ROOT / ENSEMBLE_DIR_NAME

def load_spatial_probs(model_dir, fold, image):
    path = OUTPUT_ROOT / model_dir / fold / f"{image}_spatial_probs.npy"
    return np.load(path) if path.exists() else None

def load_gt(ref_model_dir, fold, image):
    path = OUTPUT_ROOT / ref_model_dir / fold / f"{image}_gt.npy"
    if not path.exists(): return None, None
    gt = np.load(path)
    mask_labeled = gt > 0
    gt_shifted = np.zeros_like(gt)
    gt_shifted[mask_labeled] = gt[mask_labeled] - 1
    return gt_shifted, mask_labeled

def get_model_weights(models_dirs):
    return np.ones(len(models_dirs)) / len(models_dirs)

# =============================================================================
# MODULO 2: HIERARCHICAL OPTIMIZER
# =============================================================================
class HierarchicalOptimizer:
    def __init__(self):
        self.best_tau_bv = 0.1
        self.best_tau_tt = 0.3

    def fit(self, y_true, y_probs):
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
        mask_bv = prob_map[..., ID_BV] >= self.best_tau_bv
        final_map[mask_bv] = ID_BV
        mask_tt = (prob_map[..., ID_TT] >= self.best_tau_tt) & (~mask_bv)
        final_map[mask_tt] = ID_TT
        return final_map

# =============================================================================
# MODULO 3: STACKING
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
# MAIN
# =============================================================================
def main():
    print("🚀 Starting Advanced Hierarchical Ensemble V2 (Includes Soft Voting)...")
    ensure_dir(ENSEMBLE_PATH)

    model_weights = get_model_weights(MODELS_DIRS)
    ref_path = OUTPUT_ROOT / MODELS_DIRS[0]
    folds = sorted([d.name for d in ref_path.glob("fold_*")])

    stacker = ClinicalStackingV2()
    th_optimizer = HierarchicalOptimizer()
    metrics_log = []

    for test_fold in folds:
        print(f"\n🔹 FOLD TEST: {test_fold}")
        train_folds = [f for f in folds if f != test_fold]

        # A. TRAINING DATA
        X_train_list = []
        y_train_list = []

        print("  ⏳ Caricamento dati training...")
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
                indices_bg = valid_indices[gt_flat[valid_indices] == ID_BG]

                rng = np.random.default_rng(42)
                keep_nt = rng.choice(indices_nt, size=int(len(indices_nt)*0.10), replace=False) if len(indices_nt) > 0 else []
                keep_bg = rng.choice(indices_bg, size=int(len(indices_bg)*0.05), replace=False) if len(indices_bg) > 0 else []

                final_indices = np.concatenate([indices_bv, indices_tt, keep_nt, keep_bg]).astype(int)
                if len(final_indices) == 0: continue

                img_probs = []
                valid_load = True
                for m_dir in MODELS_DIRS:
                    p = load_spatial_probs(m_dir, t_fold, img)
                    if p is None: valid_load = False; break
                    img_probs.append(p.reshape(-1, 4)[final_indices])

                if valid_load:
                    stack_features = np.concatenate(img_probs, axis=1)
                    X_train_list.append(stack_features)
                    y_train_list.append(gt_flat[final_indices])

        if not X_train_list: continue
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        print(f"  🧠 Training Stacker V2 & Thresholds su {len(y_train)} pixel...")
        stacker.fit(X_train, y_train)

        # Train Thresholds
        n_samples = X_train.shape[0]
        n_models = len(MODELS_DIRS)
        X_reshaped = X_train.reshape(n_samples, n_models, 4)
        w_broad = model_weights.reshape(1, n_models, 1)
        y_probs_soft_train = np.sum(X_reshaped * w_broad, axis=1)
        th_optimizer.fit(y_train, y_probs_soft_train)

        del X_train, y_train, X_train_list, y_train_list
        gc.collect()

        # C. INFERENCE
        ensure_dir(ENSEMBLE_PATH / test_fold)
        test_images = [f.name.replace("_spatial_probs.npy", "") for f in (ref_path / test_fold).glob("*_spatial_probs.npy")]

        for img in test_images:
            gt_map, mask_labeled = load_gt(MODELS_DIRS[0], test_fold, img)
            if gt_map is None: continue

            probs_list = []
            for m_dir in MODELS_DIRS:
                probs_list.append(load_spatial_probs(m_dir, test_fold, img))
            probs_arr = np.array(probs_list)

            # 1. Soft Voting Calculation
            soft_vote_map = np.zeros_like(probs_arr[0])
            for i in range(len(model_weights)):
                soft_vote_map += probs_arr[i] * model_weights[i]

            # Predizione Soft Voting (Argmax classico)
            pred_soft = np.argmax(soft_vote_map, axis=-1)

            # 2. Hierarchical Thresholding
            pred_thresh = th_optimizer.predict(soft_vote_map)

            # 3. Stacking
            stack_input = np.concatenate(probs_list, axis=-1)
            pred_stack = stacker.predict(stack_input)

            # Save
            np.save(ENSEMBLE_PATH / test_fold / f"{img}_soft_voting.npy", pred_soft)
            np.save(ENSEMBLE_PATH / test_fold / f"{img}_hierarchical_thresh.npy", pred_thresh)
            np.save(ENSEMBLE_PATH / test_fold / f"{img}_stacking_v2.npy", pred_stack)

            # Evaluate
            mask_eval = mask_labeled & (gt_map != ID_BG)
            if np.sum(mask_eval) > 0:
                y_true_eval = gt_map[mask_eval]

                # Helper per valutazione
                def get_masked_pred(pred_map):
                    p = pred_map[mask_eval].astype(np.int32)
                    p[p == ID_BG] = -1
                    return p

                y_pred_soft = get_masked_pred(pred_soft)
                y_pred_th = get_masked_pred(pred_thresh)
                y_pred_st = get_masked_pred(pred_stack)

                m_soft = compute_all_metrics(y_true_eval, y_pred_soft, num_classes=3, labels=[0,1,2])
                m_th = compute_all_metrics(y_true_eval, y_pred_th, num_classes=3, labels=[0,1,2])
                m_st = compute_all_metrics(y_true_eval, y_pred_st, num_classes=3, labels=[0,1,2])

                row = {"fold": test_fold, "image": img}
                for k, v in m_soft.items(): row[f"soft_{k}"] = v
                for k, v in m_th.items(): row[f"thresh_{k}"] = v
                for k, v in m_st.items(): row[f"stack_v2_{k}"] = v
                metrics_log.append(row)

    if metrics_log:
        df = pd.DataFrame(metrics_log)
        out_csv = ENSEMBLE_PATH / "stacking_v2_metrics.csv"
        df.to_csv(out_csv, index=False)
        print("\n--- RISULTATI FINALI ---")
        print(df[["soft_f1_macro", "thresh_f1_macro", "stack_v2_f1_macro"]].mean().to_string())

if __name__ == "__main__":
    main()
