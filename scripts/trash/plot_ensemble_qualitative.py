import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

# --- CONFIGURAZIONE ---
PATIENTS_TO_PLOT = ["012-01", "042-02", "058-02"]

# --- PERCORSI ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# --- CONFIGURAZIONE COLORI (SLIDE STYLE) ---
# Mappatura: 0=Bianco, 1=Verde, 2=Rosso, 3=Blu, 4=Nero
cmap_dict = {
    0: '#FFFFFF',  # Unlabeled -> Bianco
    1: '#00FF00',  # NT -> Verde Lime (Standard Slide)
    2: '#FF0000',  # TT -> Rosso Puro
    3: '#0000FF',  # BV -> Blu Puro
    4: '#000000'   # BG -> Nero
}
colors_list = [cmap_dict[i] for i in range(5)]
CMAP_LABELS = mcolors.ListedColormap(colors_list)
# Normalizzazione fissa per garantire che il valore X abbia sempre lo stesso colore
NORM_LABELS = mcolors.Normalize(vmin=0, vmax=4)

def find_latest_ensemble_dir():
    dirs = sorted(list(OUTPUTS_DIR.glob("ensemble_*")))
    return dirs[-1] if dirs else None

def create_natural_rgb(cube):
    """
    Crea una RGB naturale cercando le lunghezze d'onda fisiche.
    Dataset: 400-1000nm, 128 bande.
    Step: ~4.7 nm/banda.
    """
    bands, H, W = cube.shape
    
    # Indici precisi basati sulla fisica (approssimati per il dataset 128 bande)
    # Red (640nm)   -> (640-400)/4.7 ~= Banda 51
    # Green (550nm) -> (550-400)/4.7 ~= Banda 32
    # Blue (470nm)  -> (470-400)/4.7 ~= Banda 15
    
    idx_r = 51 if bands > 51 else bands // 2
    idx_g = 32 if bands > 32 else bands // 3
    idx_b = 15 if bands > 15 else bands // 4
    
    r = cube[idx_r, :, :]
    g = cube[idx_g, :, :]
    b = cube[idx_b, :, :]
    
    rgb = np.stack([r, g, b], axis=-1)
    
    # Normalizzazione "Soft" (senza tagliare troppo i picchi)
    # Questo mantiene l'aspetto "carnoso" e un po' scuro della realtà
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-8)
    
    # Nessuna gamma correction aggressiva (gamma=1.0 o lieve schiarita 0.9)
    # Se l'immagine è troppo scura, abbassa gamma a 0.8. Se troppo chiara, alza a 1.2.
    gamma = 0.9 
    rgb = np.power(rgb, gamma)
    
    return np.clip(rgb, 0, 1)

def main():
    ensemble_dir = find_latest_ensemble_dir()
    if not ensemble_dir:
        print("❌ Nessuna cartella ensemble trovata.")
        return
    print(f"📂 Generazione plot (Stile Slide) da: {ensemble_dir.name}")
    
    save_dir = PROJECT_ROOT / "figures" / "ensemble_visuals_final"
    save_dir.mkdir(parents=True, exist_ok=True)

    for patient_id in PATIENTS_TO_PLOT:
        print(f"Processing {patient_id}...")
        
        # 1. Trova il fold
        found = False
        fold_name = ""
        for f_path in (ensemble_dir / "majority_voting").glob("fold_*"):
            if (f_path / f"{patient_id}_spectral_pred.npy").exists():
                fold_name = f_path.name
                found = True
                break
        
        if not found: continue

        try:
            # 2. Carica Dati
            cube = np.load(DATA_DIR / patient_id / "preprocessed_cube.npy")
            rgb_img = create_natural_rgb(cube)

            # Carica GT (0=Unlabeled, 1=NT, 2=TT, 3=BV, 4=BG)
            gt = np.load(ensemble_dir / "majority_voting" / fold_name / f"{patient_id}_gt.npy")
            
            # Carica Predizioni (0=NT, 1=TT, 2=BV, 3=BG)
            mv_raw = np.load(ensemble_dir / "majority_voting" / fold_name / f"{patient_id}_spectral_pred.npy")
            or_raw = np.load(ensemble_dir / "tumor_or" / fold_name / f"{patient_id}_spectral_pred.npy")
            disagreement = np.load(ensemble_dir / "disagreement" / fold_name / f"{patient_id}_disagreement.npy")

            # 3. Allinea Predizioni ai Colori del GT
            # Predizione 0 (NT) -> deve diventare 1 (Verde)
            # Predizione 3 (BG) -> deve diventare 4 (Nero)
            mv_plot = mv_raw + 1
            or_plot = or_raw + 1

        except Exception as e:
            print(f"  ❌ Errore: {e}")
            continue

        # --- PLOT ---
        fig, axes = plt.subplots(1, 5, figsize=(24, 6))
        
        # A. Pseudo-RGB Naturale
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"{patient_id}\nPseudo-RGB", fontsize=14)
        axes[0].axis('off')
        
        # B. Ground Truth (Human)
        axes[1].imshow(gt, cmap=CMAP_LABELS, norm=NORM_LABELS, interpolation='nearest')
        axes[1].set_title("Ground Truth\n(Human)", fontsize=14)
        axes[1].axis('off')
        
        # C. Majority Voting
        axes[2].imshow(mv_plot, cmap=CMAP_LABELS, norm=NORM_LABELS, interpolation='nearest')
        axes[2].set_title("Ensemble (Majority)\nPrecision Orientated", fontsize=14)
        axes[2].axis('off')

        # D. Tumor OR
        axes[3].imshow(or_plot, cmap=CMAP_LABELS, norm=NORM_LABELS, interpolation='nearest')
        axes[3].set_title("Ensemble (OR)\nRecall Orientated", fontsize=14)
        axes[3].axis('off')
        
        # E. Disagreement
        # Usiamo 'magma' o 'inferno' per l'incertezza, come nelle heatmap standard
        heatmap = axes[4].imshow(disagreement, cmap='magma', interpolation='nearest')
        axes[4].set_title("Uncertainty Map\n(Disagreement)", fontsize=14)
        axes[4].axis('off')
        plt.colorbar(heatmap, ax=axes[4], fraction=0.046, pad=0.04)

        # Legenda in basso
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFFFFF', edgecolor='gray', label='Unlabeled'),
            Patch(facecolor='#00FF00', label='NT (Sano)'),
            Patch(facecolor='#FF0000', label='TT (Tumore)'),
            Patch(facecolor='#0000FF', label='BV (Vaso)'),
            Patch(facecolor='#000000', label='BG (Sfondo)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02), fontsize=14)

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
        out_path = save_dir / f"{patient_id}_final_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  ✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
