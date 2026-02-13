import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from datetime import datetime
import shutil
import sys

# --- CONFIGURAZIONE PERCORSI ---

# 1. Trova la posizione assoluta di questo script (run_ensemble.py)
SCRIPT_DIR = Path(__file__).resolve().parent  # Es: /home/ale/r/H/scripts

# 2. Risali di un livello per trovare la Root del progetto
PROJECT_ROOT = SCRIPT_DIR.parent              # Es: /home/ale/r/H

# 3. Punta alla cartella outputs corretta
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

print(f"📍 Project Root rilevata: {PROJECT_ROOT}")
print(f"📂 Cartella Outputs impostata su: {OUTPUT_ROOT}")

# Verifica che la cartella outputs esista davvero
if not OUTPUT_ROOT.exists():
    print(f"❌ ERRORE CRITICO: La cartella {OUTPUT_ROOT} non esiste!")
    sys.exit(1)

# --- CONFIGURAZIONE MODELLI ---

MODELS_DIRS = [
    "svm-l_20251116_181835",
    "random_20251115_183848",
    "rf_20251112_162132",
    "knn-c_20251112_152155",
    "dnn_20251111_221604",
    "svm-rbf_20251112_041304"
]

# Mapping delle classi
NT_ID = 0
TT_ID = 1 
BV_ID = 2
BG_ID = 3

ENSEMBLE_DIR_NAME = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
ENSEMBLE_PATH = OUTPUT_ROOT / ENSEMBLE_DIR_NAME


# --- FUNZIONI HELPER ---

def load_prediction_map(model_dir, fold_name, image_name):
    """Carica la predizione spettrale per un modello specifico."""
    path = OUTPUT_ROOT / model_dir / fold_name / f"{image_name}_spectral_pred.npy"
    if path.exists():
        return np.load(path)
    return None

def load_gt_map(model_dir, fold_name, image_name):
    """Carica il Ground Truth (basta prenderlo dal primo modello trovato)."""
    path = OUTPUT_ROOT / model_dir / fold_name / f"{image_name}_gt.npy"
    if path.exists():
        return np.load(path)
    return None

def calculate_metrics(gt_map, pred_map):
    """
    Calcola metriche base ignorando BG e Unlabeled.
    Adatta la logica di _evaluate_map del tuo train.py.
    """
    # Flatten
    y_true = gt_map.ravel()
    y_pred = pred_map.ravel()
    
    # Maschera: GT deve essere etichettato (>0 se raw, ma qui assumiamo che il GT
    # salvato nelle cartelle fold_X sia già processato. 
    # ATTENZIONE: Solitamente il GT salvato è raw (0=Unlabeled, 1=NT, 2=TT...)
    # I modelli predicono 0=NT, 1=TT (0-indexed).
    # Dobbiamo allineare.
    
    # Assunzione: GT file ha 0=Unlabeled, 1=NT, 2=TT, 3=BV, 4=BG
    # Pred file ha 0=NT, 1=TT, 2=BV, 3=BG
    
    mask = (y_true > 0) & (y_true != 4) # Ignora Unlabeled(0) e BG(4 nel GT originale)
    
    if not np.any(mask):
        return {}

    y_t = y_true[mask] - 1 # Shift GT to 0-indexed (1->0, 2->1...)
    y_p = y_pred[mask]
    
    # Binary F1 per Tumore (Classe 1 dopo lo shift)
    # TT è 1 sia nel GT shiftato che nella predizione
    y_t_bin = (y_t == TT_ID).astype(int)
    y_p_bin = (y_p == TT_ID).astype(int)
    
    return {
        "macro_f1": f1_score(y_t, y_p, average='macro', zero_division=0),
        "tumor_f1": f1_score(y_t_bin, y_p_bin, zero_division=0),
        "tumor_recall": recall_score(y_t_bin, y_p_bin, zero_division=0),     # Sensitivity
        "tumor_precision": precision_score(y_t_bin, y_p_bin, zero_division=0) # Specificity approx
    }

# --- MAIN LOOP ---

def main():
    print(f"🚀 Avvio Ensemble. Modelli: {MODELS_DIRS}")
    print(f"📂 Output directory: {ENSEMBLE_PATH}")
    
    metrics_log = []
    
    # Crea la struttura delle cartelle
    variants = ["majority_voting", "tumor_or", "tumor_and", "disagreement"]
    for v in variants:
        (ENSEMBLE_PATH / v).mkdir(parents=True, exist_ok=True)

    # 1. Identifica i fold e le immagini comuni
    # Prendiamo il primo modello come riferimento per la lista file
    ref_model = MODELS_DIRS[0]
    ref_path = OUTPUT_ROOT / ref_model
    
    for fold_dir in sorted(ref_path.glob("fold_*_predictions")):
        fold_name = fold_dir.name
        print(f"\nProcessing {fold_name}...")
        
        # Prepara sottocartelle per questo fold in ogni variante
        for v in variants:
            (ENSEMBLE_PATH / v / fold_name).mkdir(exist_ok=True)
            
        pred_files = list(fold_dir.glob("*_spectral_pred.npy"))
        
        for p_file in pred_files:
            image_name = p_file.name.replace("_spectral_pred.npy", "")
            
            # --- FASE 1: RACCOLTA PREDIZIONI ---
            stack_list = []
            valid_image = True
            
            for m_dir in MODELS_DIRS:
                p_map = load_prediction_map(m_dir, fold_name, image_name)
                if p_map is None:
                    print(f"⚠️ Missing prediction for {image_name} in {m_dir}. Skipping image.")
                    valid_image = False
                    break
                stack_list.append(p_map)
                
            if not valid_image:
                continue
                
            # Stack shape: (N_Models, H, W)
            stack = np.array(stack_list)
            
            # Carica GT per metriche
            gt_map = load_gt_map(ref_model, fold_name, image_name)
            
            # --- FASE 2: CALCOLO VARIANTI ---
            
            # A. MAJORITY VOTING (MV)
            # mode restituisce [valori_modali], [conteggi]
            mv_res = stats.mode(stack, axis=0, keepdims=False)
            mv_map = mv_res.mode.astype(np.uint8)
            agreement_count = mv_res.count # Quanti modelli hanno votato per la moda
            
            # B1. TUMOR OR (Union - High Sensitivity)
            # Se ALMENO UN modello dice TT (1), diventa TT. Altrimenti resta MV.
            tumor_mask_any = np.any(stack == TT_ID, axis=0)
            or_map = mv_map.copy()
            or_map[tumor_mask_any] = TT_ID
            
            # B2. TUMOR AND (Intersection - High Specificity)
            # Se TUTTI i modelli dicono TT, è TT.
            # Se MV dice TT, ma NON TUTTI sono d'accordo, facciamo fallback a NT (0).
            # (Riduciamo i falsi positivi ai bordi)
            tumor_mask_all = np.all(stack == TT_ID, axis=0)
            and_map = mv_map.copy()
            # Dove MV dice tumore, ma non c'è unanimità -> forza a NT
            mask_uncertain_tumor = (mv_map == TT_ID) & (~tumor_mask_all)
            and_map[mask_uncertain_tumor] = NT_ID 
            
            # C. DISAGREEMENT MAP (Uncertainty)
            # Calcoliamo quanti modelli NON sono d'accordo col voto finale (MV)
            n_models = len(MODELS_DIRS)
            disagreement_map = (n_models - agreement_count).astype(np.uint8)
            
            # --- FASE 3: SALVATAGGIO & METRICHE ---
            
            # Salva mappe
            base_out = ENSEMBLE_PATH
            np.save(base_out / "majority_voting" / fold_name / f"{image_name}_spectral_pred.npy", mv_map)
            np.save(base_out / "tumor_or" / fold_name / f"{image_name}_spectral_pred.npy", or_map)
            np.save(base_out / "tumor_and" / fold_name / f"{image_name}_spectral_pred.npy", and_map)
            np.save(base_out / "disagreement" / fold_name / f"{image_name}_disagreement.npy", disagreement_map)
            
            # Copia anche il GT per comodità in ogni cartella (per compatibilità con script di visualizzazione futuri)
            for v in variants:
                 if v != "disagreement": # Disagreement non è una label map classica
                     np.save(base_out / v / fold_name / f"{image_name}_gt.npy", gt_map)

            # Calcola metriche
            if gt_map is not None:
                row = {"image": image_name, "fold": fold_name}
                
                # Metriche per MV
                m_mv = calculate_metrics(gt_map, mv_map)
                for k, val in m_mv.items(): row[f"mv_{k}"] = val
                
                # Metriche per OR
                m_or = calculate_metrics(gt_map, or_map)
                for k, val in m_or.items(): row[f"or_{k}"] = val
                
                # Metriche per AND
                m_and = calculate_metrics(gt_map, and_map)
                for k, val in m_and.items(): row[f"and_{k}"] = val
                
                metrics_log.append(row)

    # --- FASE 4: REPORT FINALE ---
    if metrics_log:
        df = pd.DataFrame(metrics_log)
        csv_path = ENSEMBLE_PATH / "ensemble_metrics_comparison.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n--- RISULTATI MEDI (Ensemble vs Varianti) ---")
        cols = ["mv_tumor_f1", "or_tumor_f1", "and_tumor_f1", 
                "mv_tumor_recall", "or_tumor_recall", # Sensitivity
                "mv_tumor_precision", "and_tumor_precision"] # Precision/Specificity
        
        summary = df[cols].mean()
        print(summary.to_string())
        print(f"\nReport completo salvato in: {csv_path}")
        print(f"Mappe salvate in: {ENSEMBLE_PATH}")

if __name__ == "__main__":
    main()
