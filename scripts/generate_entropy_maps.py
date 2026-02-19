"""
generate_entropy_maps.py
------------------------
Genera Heatmap di Incertezza e analizza la correlazione tra Incertezza ed Errore.
Implementa la "Predictive Uncertainty Decomposition":
1. Total Uncertainty (Entropy of Mean)
2. Aleatoric Uncertainty (Mean of Entropies)
3. Epistemic Uncertainty (Mutual Information)

Output:
- 3 cartelle di immagini PNG per ogni tipo di incertezza.
- CSV con statistiche.
- Cartella 'top_correlated' con analisi approfondita (4 mappe) per le 5 immagini
  con la correlazione media più alta.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pointbiserialr
import sys
import shutil

# --- CONFIGURAZIONE ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures" / "uncertainty_maps"

# LISTA COMPLETA DEI 6 MODELLI PER L'ENSEMBLE
MODELS_DIRS = [
    "rf_20260216_001014",
    "svm-l_20260216_004446",
    "svm-rbf_20260216_015137",
    "dnn_20260216_033845",
    "knn-c_20260216_040940",
    "knn-e_20260216_033918"
]

def load_spatial_probs(model_dir, fold, image):
    path = OUTPUTS_DIR / model_dir / fold / f"{image}_spatial_probs.npy"
    return np.load(path) if path.exists() else None

def load_gt_raw(ref_model_dir, fold, image):
    path = OUTPUTS_DIR / ref_model_dir / fold / f"{image}_gt.npy"
    return np.load(path) if path.exists() else None

def fast_entropy(probs, base=4):
    """
    Calcola l'entropia di Shannon in modo vettorializzato.
    """
    # Clip per evitare log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    ent = -np.sum(probs * np.log(probs), axis=-1) / np.log(base)
    return ent

def compute_uncertainty_decomposition(probs_list):
    """
    Calcola le tre componenti dell'incertezza.
    Returns: u_total, u_aleatoric, u_epistemic (Tutte HxW)
    """
    stacked_probs = np.array(probs_list)

    # 1. Media (Soft Voting) -> p_mean
    p_mean = np.mean(stacked_probs, axis=0)

    # 2. Total Uncertainty: H(p_mean)
    u_total = fast_entropy(p_mean, base=4)

    # 3. Aleatoric Uncertainty: Mean(H(p_i))
    all_entropies = fast_entropy(stacked_probs, base=4)
    u_aleatoric = np.mean(all_entropies, axis=0)

    # 4. Epistemic Uncertainty: Total - Aleatoric
    u_epistemic = u_total - u_aleatoric
    u_epistemic = np.maximum(u_epistemic, 0.0) # Fix floating point errors

    return u_total, u_aleatoric, u_epistemic

def save_single_map(data, mask, folder, img_name, title, cmap='magma'):
    """Helper per salvare una singola mappa mascherata"""
    viz_data = data.copy()
    viz_data[~mask] = 0.0

    plt.figure(figsize=(10, 8))
    plt.imshow(viz_data, cmap=cmap, vmin=0, vmax=1.0)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"{title}: {img_name}", fontsize=14)
    plt.savefig(folder / f"{img_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("🚀 Generazione Mappe di Incertezza (Decomposizione) & Analisi...")

    # Setup Cartelle
    dirs = {
        "total": FIGURES_DIR / "total",
        "aleatoric": FIGURES_DIR / "aleatoric",
        "epistemic": FIGURES_DIR / "epistemic",
        "top": FIGURES_DIR / "top_correlated"
    }

    # Pulisci e ricrea
    if FIGURES_DIR.exists(): shutil.rmtree(FIGURES_DIR)
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    ref_path = OUTPUTS_DIR / MODELS_DIRS[0]
    folds = sorted([d.name for d in ref_path.glob("fold_*")])

    stats_data = []

    for fold in folds:
        print(f"\n📂 Processing {fold}...")
        images = [f.name.replace("_spatial_probs.npy", "") for f in (ref_path / fold).glob("*_spatial_probs.npy")]

        for img in images:
            # Load Probs
            probs_list = []
            valid = True
            for m_dir in MODELS_DIRS:
                p = load_spatial_probs(m_dir, fold, img)
                if p is None: valid = False; break
                probs_list.append(p)

            if not valid: continue

            # Load GT
            gt = load_gt_raw(MODELS_DIRS[0], fold, img)
            if gt is None: continue

            # --- 1. CALCOLO LE 3 INCERTEZZE ---
            u_tot, u_ale, u_epi = compute_uncertainty_decomposition(probs_list)

            # Calcolo Errore (basato su Soft Voting)
            soft_vote_map = np.mean(np.array(probs_list), axis=0)
            pred_map = np.argmax(soft_vote_map, axis=-1)

            # Mask Valid (No BG/Unlabeled)
            mask_valid = (gt != 0) & (gt != 4)

            # Mappa Errore (1=Errore)
            error_map = np.zeros_like(gt, dtype=float)
            error_map[mask_valid] = ( (pred_map[mask_valid] + 1) != gt[mask_valid] ).astype(float)

            # --- 2. SALVATAGGIO MAPPE INDIVIDUALI ---
            save_single_map(u_tot, mask_valid, dirs['total'], img, "Total Uncertainty")
            save_single_map(u_ale, mask_valid, dirs['aleatoric'], img, "Aleatoric Uncertainty")
            save_single_map(u_epi, mask_valid, dirs['epistemic'], img, "Epistemic Uncertainty")

            # --- 3. CALCOLO CORRELAZIONI ---
            # Solo su pixel validi
            u_tot_valid = u_tot[mask_valid]
            u_ale_valid = u_ale[mask_valid]
            u_epi_valid = u_epi[mask_valid]
            err_valid = error_map[mask_valid]

            corr_avg = -1.0

            if len(err_valid) > 100 and np.var(err_valid) > 0:
                c_t, _ = pointbiserialr(err_valid, u_tot_valid)
                c_a, _ = pointbiserialr(err_valid, u_ale_valid)
                c_e, _ = pointbiserialr(err_valid, u_epi_valid)

                # Gestione NaN
                c_t = 0 if np.isnan(c_t) else c_t
                c_a = 0 if np.isnan(c_a) else c_a
                c_e = 0 if np.isnan(c_e) else c_e

                corr_avg = (c_t + c_a + c_e) / 3.0

                print(f"   -> {img}: Avg Corr={corr_avg:.3f} (Tot={c_t:.2f}, Ale={c_a:.2f}, Epi={c_e:.2f})")

                # Salva per ranking (Manteniamo le mappe mascherate in memoria per i top 5)
                stats_data.append({
                    "image": img,
                    "corr_avg": corr_avg,
                    "corr_total": c_t,
                    "corr_aleatoric": c_a,
                    "corr_epistemic": c_e,
                    "u_tot": u_tot * mask_valid,
                    "u_ale": u_ale * mask_valid,
                    "u_epi": u_epi * mask_valid,
                    "err": error_map,
                    "mask": mask_valid
                })

    # --- 4. SALVATAGGIO CSV ---
    print("\n💾 Salvataggio CSV riassuntivo...")
    df_stats = pd.DataFrame(stats_data)
    cols_to_save = ["image", "corr_avg", "corr_total", "corr_aleatoric", "corr_epistemic"]
    if not df_stats.empty:
        df_stats[cols_to_save].to_csv(OUTPUTS_DIR / "uncertainty_correlations.csv", index=False)

        # --- 5. TOP 5 COMPOSITE PLOT ---
        print("\n🏆 Generazione Top 5 Immagini (Basato su Media Correlazioni)...")
        top_5 = df_stats.sort_values(by="corr_avg", ascending=False).head(5)

        for idx, row in top_5.iterrows():
            img_name = row['image']
            print(f"   Generazione plot per: {img_name} (Avg Corr: {row['corr_avg']:.3f})")

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Helper plot
            def plot_sub(ax, data, title, is_error=False):
                if is_error:
                    # Custom error map (Rosso su Grigio)
                    viz = np.zeros(data.shape + (3,))
                    mask = row['mask']
                    # Sfondo grigio dove c'è tessuto (mask_valid) ma non errore
                    viz[(data == 0) & mask] = [0.2, 0.2, 0.2]
                    # Rosso dove c'è errore
                    viz[data == 1] = [1, 0, 0]
                    # Nero altrove (BG)
                    ax.imshow(viz)
                else:
                    im = ax.imshow(data, cmap='magma', vmin=0, vmax=1.0)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.set_title(title, fontsize=12)
                ax.axis('off')

            plot_sub(axes[0,0], row['u_tot'], f"Total Uncertainty\nCorr: {row['corr_total']:.2f}")
            plot_sub(axes[0,1], row['u_ale'], f"Aleatoric Uncertainty\nCorr: {row['corr_aleatoric']:.2f}")
            plot_sub(axes[1,0], row['u_epi'], f"Epistemic Uncertainty\nCorr: {row['corr_epistemic']:.2f}")
            plot_sub(axes[1,1], row['err'],   f"Misclassification (Red)\n(Ground Truth Mismatch)", is_error=True)

            plt.suptitle(f"Uncertainty Analysis: {img_name}", fontsize=16)
            plt.tight_layout()
            plt.savefig(dirs['top'] / f"top_{img_name}_analysis.png", dpi=150)
            plt.close()

    print(f"\n✅ Finito! Controlla: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
