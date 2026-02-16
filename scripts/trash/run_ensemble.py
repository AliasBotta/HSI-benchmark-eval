import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import sys
from datetime import datetime

# --- CONFIGURAZIONE PERCORSI E IMPORT ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Importa le metriche originali
from utils.metrics import compute_all_metrics
from utils.helpers import ensure_dir

# --- CONFIGURAZIONE ---
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MODELS_DIRS = [
    "svm-l_20251116_181835",
    "rf_20260213_183145",
    "knn-c_20260213_190830",
    "dnn_20260213_185058",
    "svm-rbf_20260213_195703"
]

# ID Classi (Modelli: 0-based, GT: 1-based)
TT_ID_PRED = 1  # Tumor Tissue nelle predizioni (0=NT, 1=TT)
TT_ID_GT = 2    # Tumor Tissue nel GT (1=NT, 2=TT)

# Crea nome output
ENSEMBLE_DIR_NAME = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
ENSEMBLE_PATH = OUTPUT_ROOT / ENSEMBLE_DIR_NAME

# --- FUNZIONI HELPER ---

def load_prediction_map(model_dir, fold_name, image_name):
    path = OUTPUT_ROOT / model_dir / fold_name / f"{image_name}_spectral_pred.npy"
    if path.exists():
        return np.load(path)
    return None

def load_gt_map(model_dir, fold_name, image_name):
    path = OUTPUT_ROOT / model_dir / fold_name / f"{image_name}_gt.npy"
    if path.exists():
        return np.load(path)
    return None

def evaluate_map_compliant(gt_map, pred_map):
    """
    Calcola le metriche ESATTAMENTE come train.py.
    """
    gt = gt_map.flatten()
    pr = pred_map.flatten()

    # 1. Filtra Unlabeled (GT=0)
    mask_labeled = gt > 0
    gt = gt[mask_labeled] - 1  # Shift a 0-based
    pr = pr[mask_labeled]

    # 2. Filtra Background dal GT (GT raw era 4 -> ora è 3)
    # Nota: 0=NT, 1=TT, 2=BV, 3=BG
    keep = gt != 3 
    gt = gt[keep]
    pr = pr[keep]

    # 3. Gestisci predizioni BG come errori (-1)
    pr_eval = np.where(pr > 2, -1, pr)

    # 4. Calcola metriche (0=NT, 1=TT, 2=BV)
    return compute_all_metrics(gt, pr_eval, num_classes=3, labels=[0, 1, 2])

def flatten_metrics(metrics_dict, prefix):
    row = {}
    for k, v in metrics_dict.items():
        row[f"{prefix}_{k}"] = v
    return row

# --- MAIN LOOP ---

def main():
    print(f"🚀 Avvio Ensemble (Strict Mode). Modelli: {len(MODELS_DIRS)}")
    
    if not OUTPUT_ROOT.exists():
        print(f"❌ Errore: {OUTPUT_ROOT} non esiste.")
        return

    variants = ["majority_voting", "tumor_or", "tumor_and", "disagreement"]
    for v in variants:
        ensure_dir(ENSEMBLE_PATH / v)

    metrics_log = []
    
    ref_model = MODELS_DIRS[0]
    ref_path = OUTPUT_ROOT / ref_model
    
    for fold_dir in sorted(ref_path.glob("fold_*_predictions")):
        fold_name = fold_dir.name
        print(f"\nProcessing {fold_name}...")
        
        for v in variants:
            ensure_dir(ENSEMBLE_PATH / v / fold_name)
            
        pred_files = list(fold_dir.glob("*_spectral_pred.npy"))
        
        for p_file in pred_files:
            image_name = p_file.name.replace("_spectral_pred.npy", "")
            
            # 1. Carica GT subito per verificare validità
            gt_map = load_gt_map(ref_model, fold_name, image_name)
            if gt_map is None:
                continue

            # --- FILTRO CRUCIALE (COMPLIANCE TESI) ---
            # Se nel GT non c'è tumore (Classe 2), saltiamo l'immagine.
            # Inserire immagini senza tumore abbassa la media della sensitivity a caso.
            if not np.any(gt_map == TT_ID_GT):
                # print(f"  ⏭️ Skipping {image_name}: No Tumor in GT (Compliant).")
                continue
            
            # 2. Carica Predizioni
            stack_list = []
            valid_image = True
            for m_dir in MODELS_DIRS:
                p_map = load_prediction_map(m_dir, fold_name, image_name)
                if p_map is None:
                    valid_image = False
                    break
                stack_list.append(p_map)
                
            if not valid_image:
                continue
            
            stack = np.array(stack_list)

            # 3. Calcolo Ensemble
            
            # A. Majority Voting
            mv_res = stats.mode(stack, axis=0, keepdims=False)
            mv_map = mv_res.mode.astype(np.uint8)
            agreement = mv_res.count
            
            # B. Tumor OR (TT_ID_PRED = 1)
            tumor_mask_any = np.any(stack == TT_ID_PRED, axis=0)
            or_map = mv_map.copy()
            or_map[tumor_mask_any] = TT_ID_PRED
            
            # C. Tumor AND
            tumor_mask_all = np.all(stack == TT_ID_PRED, axis=0)
            and_map = mv_map.copy()
            mask_uncertain = (mv_map == TT_ID_PRED) & (~tumor_mask_all)
            and_map[mask_uncertain] = 0 # Fallback a NT
            
            # D. Disagreement
            n_models = len(MODELS_DIRS)
            disagreement_map = (n_models - agreement).astype(np.uint8)
            
            # 4. Salvataggio
            base_out = ENSEMBLE_PATH
            np.save(base_out / "majority_voting" / fold_name / f"{image_name}_spectral_pred.npy", mv_map)
            np.save(base_out / "tumor_or" / fold_name / f"{image_name}_spectral_pred.npy", or_map)
            np.save(base_out / "tumor_and" / fold_name / f"{image_name}_spectral_pred.npy", and_map)
            np.save(base_out / "disagreement" / fold_name / f"{image_name}_disagreement.npy", disagreement_map)
            
            for v in variants:
                if v != "disagreement":
                    np.save(base_out / v / fold_name / f"{image_name}_gt.npy", gt_map)
            
            # 5. Metriche
            row = {"fold": fold_name, "image": image_name}
            
            m_mv = evaluate_map_compliant(gt_map, mv_map)
            m_or = evaluate_map_compliant(gt_map, or_map)
            m_and = evaluate_map_compliant(gt_map, and_map)
            
            row.update(flatten_metrics(m_mv, "mv"))
            row.update(flatten_metrics(m_or, "or"))
            row.update(flatten_metrics(m_and, "and"))
            
            metrics_log.append(row)

    # 6. Report
    if metrics_log:
        df = pd.DataFrame(metrics_log)
        csv_path = ENSEMBLE_PATH / "ensemble_metrics_compliant.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n--- RISULTATI MEDI (SOLO SU IMMAGINI CON TUMORE) ---")
        cols_summary = [
            "mv_f1_macro", "mv_sens_class_1", "mv_spec_class_1",
            "or_f1_macro", "or_sens_class_1", "or_spec_class_1",
            "and_f1_macro", "and_sens_class_1"
        ]
        
        cols_existing = [c for c in cols_summary if c in df.columns]
        summary = df[cols_existing].mean()
        
        print(summary.to_string())
        print(f"\n✅ Report salvato: {csv_path}")
        print(f"📊 Totale immagini processate: {len(df)}")

if __name__ == "__main__":
    main()
