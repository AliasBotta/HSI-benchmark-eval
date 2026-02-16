"""
plot_results_with_ensemble.py
-----------------------------
Genera grafici comparativi (Boxplot F1 e Barplot Metriche) unendo
Modelli Singoli ed Ensemble.

COLORING STRATEGY:
- Boxplot: Modelli Singoli = Uniforme (Grigio/Azzurro), Ensemble = 3 Colori Distinti.
- Barplot: Modelli Singoli = Palette Pastello, Ensemble = Stessi 3 Colori del Boxplot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

# --- CONFIGURAZIONE ---
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"

# Modelli Singoli
SINGLE_MODELS = ["rf", "knn-e", "knn-c", "svm-l", "svm-rbf", "dnn"]

# Nomi visuali per gli Ensemble
ENS_MV = "Ensemble (MV)"
ENS_OR = "Ensemble (OR)"
ENS_AND = "Ensemble (AND)"

# Ordine nel grafico
MODEL_ORDER = [m.upper() for m in SINGLE_MODELS] + [ENS_MV, ENS_OR, ENS_AND]

# --- DEFINIZIONE COLORI (Palette Coerente) ---
# Colori specifici per gli Ensemble (da usare in entrambi i grafici)
COLOR_MV  = "#1f77b4"  # Blu Scuro (Seaborn default blue)
COLOR_OR  = "#d62728"  # Rosso Scuro (Seaborn default red)
COLOR_AND = "#2ca02c"  # Verde Scuro (Seaborn default green)

# Colore unico per i modelli singoli nel BOXPLOT
COLOR_SINGLE_BOXPLOT = "#b0c4de" # LightSteelBlue (Neutro)

# --- 1. CARICAMENTO DATI ---
def load_single_models(outputs_dir):
    all_data = []
    for model_name in SINGLE_MODELS:
        run_folders = sorted(list(outputs_dir.glob(f"{model_name}_*")))
        if not run_folders: continue
        latest_run = run_folders[-1]
        summary_path = latest_run / "metrics_summary.csv"
        
        if summary_path.exists():
            try:
                df = pd.read_csv(summary_path, index_col=0)
                # Filtra solo i fold numerici
                df = df[df.index.astype(str).isin(['1', '2', '3', '4', '5'])]
                df['model'] = model_name.upper()
                df['Type'] = 'Single'
                all_data.append(df)
            except Exception: pass
    return pd.concat(all_data) if all_data else pd.DataFrame()

def load_ensemble_data(outputs_dir):
    ens_folders = sorted(list(outputs_dir.glob("ensemble_*")))
    if not ens_folders: return pd.DataFrame()
    csv_path = ens_folders[-1] / "ensemble_metrics_compliant.csv"
    if not csv_path.exists(): return pd.DataFrame()
    
    print(f"Ensemble caricato da: {ens_folders[-1].name}")
    df_raw = pd.read_csv(csv_path)
    # Aggrega per fold (media delle immagini nel fold)
    df_folds = df_raw.groupby('fold').mean(numeric_only=True).reset_index()
    
    frames = []
    strategies = [(ENS_MV, 'mv'), (ENS_OR, 'or'), (ENS_AND, 'and')]
    
    for model_name, prefix in strategies:
        df_strat = pd.DataFrame()
        # Mappa metriche globali
        df_strat['spatial_f1_macro'] = df_folds[f'{prefix}_f1_macro']
        df_strat['spatial_oa'] = df_folds[f'{prefix}_oa']
        # Mappa metriche per classe
        for c in [0, 1, 2]:
            df_strat[f'spatial_sens_class_{c}'] = df_folds[f'{prefix}_sens_class_{c}']
            df_strat[f'spatial_spec_class_{c}'] = df_folds[f'{prefix}_spec_class_{c}']
        
        df_strat['model'] = model_name
        df_strat['Type'] = 'Ensemble'
        frames.append(df_strat)
        
    return pd.concat(frames)

# --- 2. PLOT BOXPLOT F1 (Uniforme vs Colorati) ---
def plot_f1_boxplot(df, save_path):
    print("Generazione Boxplot F1...")
    
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Crea la palette custom per il Boxplot
    # Tutti i modelli singoli -> COLOR_SINGLE_BOXPLOT
    # Ensemble -> Colori specifici
    box_palette = {m: COLOR_SINGLE_BOXPLOT for m in MODEL_ORDER}
    box_palette[ENS_MV] = COLOR_MV
    box_palette[ENS_OR] = COLOR_OR
    box_palette[ENS_AND] = COLOR_AND

    ax = sns.boxplot(
        data=df,
        x='model',
        y='spatial_f1_macro',
        hue='model',       # Hue deve essere model per applicare la palette custom per-barra
        order=MODEL_ORDER,
        palette=box_palette,
        dodge=False,       # Evita che le barre si stringano
        width=0.6,
        linewidth=1.2
    )
    
    plt.title("Confronto Macro F1-Score: Modelli Singoli vs Ensemble", fontsize=18)
    plt.ylabel("Macro F1-Score (Test Set)", fontsize=14)
    plt.xlabel("")
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Linea separatrice
    idx_sep = len(SINGLE_MODELS) - 0.5
    plt.axvline(x=idx_sep, color='gray', linestyle='--', linewidth=2)
    plt.text(idx_sep + 0.1, 0.42, "Ensemble Strategies", fontsize=12, fontweight='bold', color='gray')
    
    # Rimuovi la legenda automatica (ridondante coi tick x)
    if ax.legend_: ax.legend_.remove()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# --- 3. PLOT BARPLOT DETTAGLIATO ---
def plot_metrics_barplot(df, save_path):
    print("Generazione Barplot Metriche...")
    
    metrics_map = {
        'spatial_oa': 'OA',
        'spatial_f1_macro': 'Macro F1',
        'spatial_sens_class_1': 'Sens (Tumor)', 
        'spatial_spec_class_1': 'Spec (Tumor)', 
        'spatial_sens_class_0': 'Sens (Normal)',
        'spatial_spec_class_0': 'Spec (Normal)'
    }
    
    # Preparazione dati
    cols = ['model'] + list(metrics_map.keys())
    cols = [c for c in cols if c in df.columns]
    df_melt = df[cols].melt(id_vars='model', var_name='Metric', value_name='Score')
    df_melt['Metric'] = df_melt['Metric'].map(metrics_map)
    
    metric_order = ['OA', 'Macro F1', 'Sens (Tumor)', 'Spec (Tumor)', 'Sens (Normal)', 'Spec (Normal)']

    # --- CREAZIONE PALETTE MISTA ---
    # 1. Colori distinti per i modelli singoli (Pastel)
    n_singles = len(SINGLE_MODELS)
    pastel_colors = sns.color_palette("pastel", n_singles) # Palette pastello per i singoli
    bar_palette = dict(zip([m.upper() for m in SINGLE_MODELS], pastel_colors))
    
    # 2. Colori fissi per gli Ensemble (UGUALI al Boxplot)
    bar_palette[ENS_MV] = COLOR_MV
    bar_palette[ENS_OR] = COLOR_OR
    bar_palette[ENS_AND] = COLOR_AND

    # Plot
    g = sns.catplot(
        data=df_melt,
        kind="bar",
        x="Metric",
        y="Score",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=bar_palette,  # Applica la palette mista
        height=7,
        aspect=2.2,
        errorbar='sd',
        capsize=0.04,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.95
    )
    
    g.despine(left=True)
    g.set_axis_labels("", "Score (Media ± SD)")
    g.legend.set_title("Modello / Strategia")
    g.set(ylim=(0, 1.05))
    
    # Decorazioni
    for ax in g.axes.flat:
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        # Evidenzia area tumore
        ax.axvspan(1.5, 3.5, color='gray', alpha=0.08)
        ax.text(2.5, 1.02, "AREA CRITICA (TUMORE)", ha='center', fontsize=10, fontweight='bold', color='dimgray')

    plt.subplots_adjust(top=0.92)
    g.fig.suptitle("Analisi Dettagliata: Modelli Singoli vs Strategie Ensemble", fontsize=20)
    
    g.savefig(save_path, dpi=300)
    plt.close()

# --- MAIN ---
def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    
    df_single = load_single_models(OUTPUTS_DIR)
    df_ens = load_ensemble_data(OUTPUTS_DIR)
    
    if df_single.empty or df_ens.empty:
        print("❌ Dati mancanti.")
        return
        
    df_full = pd.concat([df_single, df_ens], ignore_index=True)
    
    plot_f1_boxplot(df_full, FIGURES_DIR / "comparison_f1_ensemble_final.png")
    plot_metrics_barplot(df_full, FIGURES_DIR / "comparison_metrics_ensemble_final.png")
    
    print(f"✅ Grafici salvati in: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
