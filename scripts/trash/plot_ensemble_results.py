"""
plot_ensemble_results.py
------------------------
Genera un grafico a barre comparativo per le strategie di Ensemble.
CORREZIONE: Aggrega i dati per FOLD prima di plottare, per rendere
la Deviazione Standard (SD) confrontabile con gli esperimenti precedenti.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# --- CONFIGURAZIONE ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Palette Muted (Paper Style)
CUSTOM_PALETTE = {
    "Majority Voting": "#4c72b0",       # Deep Blue
    "Ensemble OR (Union)": "#c44e52",   # Deep Red
    "Ensemble AND (Intersection)": "#55a868" # Deep Green
}

STRATEGY_ORDER = ["Majority Voting", "Ensemble OR (Union)", "Ensemble AND (Intersection)"]

METRIC_ORDER = [
    'OA', 'Macro F1', 
    'Sens (NT)', 'Sens (TT)', 'Sens (BV)', 
    'Spec (NT)', 'Spec (TT)', 'Spec (BV)'
]

def find_latest_ensemble_dir():
    dirs = sorted(list(OUTPUTS_DIR.glob("ensemble_*")))
    return dirs[-1] if dirs else None

def get_metrics_mapping():
    mapping = {}
    strategies = {
        'mv': 'Majority Voting',
        'or': 'Ensemble OR (Union)',
        'and': 'Ensemble AND (Intersection)'
    }
    
    # Globali
    for prefix, strat_name in strategies.items():
        mapping[f'{prefix}_oa'] = (strat_name, 'OA')
        mapping[f'{prefix}_f1_macro'] = (strat_name, 'Macro F1')
        
    # Per Classe
    class_names = {0: 'NT', 1: 'TT', 2: 'BV'}
    for prefix, strat_name in strategies.items():
        for c_id, c_name in class_names.items():
            col_sens = f'{prefix}_sens_class_{c_id}'
            mapping[col_sens] = (strat_name, f'Sens ({c_name})')
            col_spec = f'{prefix}_spec_class_{c_id}'
            mapping[col_spec] = (strat_name, f'Spec ({c_name})')
            
    return mapping

def load_and_prep_data(csv_path):
    if not csv_path.exists():
        print(f"❌ File non trovato: {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # --- PASSAGGIO CHIAVE AGGIUNTO ---
    # Raggruppa per Fold e calcola la media.
    # Ora abbiamo 5 righe (una per fold), invece di 40 (una per immagine).
    # Questo riduce la varianza allo stesso livello di train.py
    df_fold_avg = df.groupby('fold').mean(numeric_only=True).reset_index()
    print(f"Dati aggregati: da {len(df)} immagini a {len(df_fold_avg)} fold.")
    # ---------------------------------

    mapping = get_metrics_mapping()
    long_data = []
    
    for _, row in df_fold_avg.iterrows():
        for col, val in row.items():
            if col in mapping:
                strategy, metric_name = mapping[col]
                long_data.append({
                    'Strategy': strategy,
                    'Metric': metric_name,
                    'Value': val
                })
                
    return pd.DataFrame(long_data)

def plot_ensemble_comparison(df_long, save_path):
    print("Generazione grafico (Aggregazione per Fold)...")
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    plt.figure(figsize=(16, 8))
    
    g = sns.barplot(
        data=df_long,
        x='Metric',
        y='Value',
        hue='Strategy',
        order=METRIC_ORDER,
        hue_order=STRATEGY_ORDER,
        palette=CUSTOM_PALETTE,
        errorbar='sd',  # Ora SD ha senso perché è "SD tra i fold"
        capsize=0.04,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.9
    )
    
    g.set_title("Performance Ensemble (Media su 5 Fold)", fontsize=18, pad=15)
    g.set_xlabel("", fontsize=0)
    g.set_ylabel("Score (Media ± SD Fold)", fontsize=14)
    g.set_ylim(0, 1.1)
    
    plt.legend(title="Strategia Ensemble", title_fontsize=12, fontsize=11, 
               loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)
    
    g.yaxis.grid(True, linestyle='--', alpha=0.7)
    g.xaxis.grid(False)
    
    plt.axvline(x=1.5, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=4.5, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafico salvato in: {save_path}")

def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    ensemble_dir = find_latest_ensemble_dir()
    if not ensemble_dir:
        print("❌ Nessuna cartella ensemble trovata.")
        return
        
    print(f"📂 Analisi dati da: {ensemble_dir.name}")
    csv_path = ensemble_dir / "ensemble_metrics_compliant.csv"
    
    df_long = load_and_prep_data(csv_path)
    
    if df_long.empty:
        print("❌ Dataset vuoto.")
        return

    output_file = FIGURES_DIR / "ensemble_strategies_fold_avg.png"
    plot_ensemble_comparison(df_long, output_file)

if __name__ == "__main__":
    main()
