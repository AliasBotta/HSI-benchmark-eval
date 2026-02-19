"""
plot_final_thesis_results.py
----------------------------
Genera il grafico conclusivo per la tesi.
Confronta:
1. Modelli Singoli (Baseline)
2. Ensemble Classico (Majority Voting)
3. Soft Voting (Miglior F1)
4. Hierarchical Thresholding (Safety)
5. Clinical Stacking V2 (AI-based Risk)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

# --- CONFIGURAZIONE ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Palette Colori
COLOR_PROPOSED = "#ff7f0e"  # Stacking V2 (Arancio)
COLOR_THRESH   = "#9467bd"  # Thresholding (Viola)
COLOR_SOFT     = "#2ca02c"  # Soft Voting (Verde)
COLOR_BASELINE = "#1f77b4"  # MV (Blu scuro)
COLOR_SINGLE   = "#aec7e8"  # Singoli (Azzurro chiaro)

SINGLE_MODELS = ["rf", "knn-c", "knn-e", "svm-l", "svm-rbf", "dnn"]
ENS_MV = "Ensemble (MV)"
ENS_SOFT = "Ensemble (Soft Voting)"
ENS_THRESH = "Hierarchical Thresh."
PROPOSED = "Clinical Stacking (V2)"

MODEL_ORDER = [m.upper() for m in SINGLE_MODELS] + [ENS_MV, ENS_SOFT, ENS_THRESH, PROPOSED]

def find_latest_run_folders(outputs_dir, model_names):
    latest = {}
    for m in model_names:
        found = sorted(list(outputs_dir.glob(f"{m}_*")))
        if found: latest[m] = found[-1]
    return latest

def load_data():
    dfs = []

    # 1. Singoli
    runs = find_latest_run_folders(OUTPUTS_DIR, SINGLE_MODELS)
    for m, path in runs.items():
        try:
            d = pd.read_csv(path / "metrics_summary.csv", index_col=0)
            d = d[d.index.astype(str).isin(['1','2','3','4','5'])]
            d['model'] = m.upper()
            d['Category'] = 'Single Model'
            dfs.append(d)
        except: pass

    # 2. Majority Voting
    ens_dirs = sorted([d for d in OUTPUTS_DIR.glob("ensemble_*") if "stacking" not in d.name])
    if ens_dirs:
        path = ens_dirs[-1] / "ensemble_metrics_compliant.csv"
        if path.exists():
            d = pd.read_csv(path)
            d_fold = d.groupby('fold').mean(numeric_only=True).reset_index()
            temp = pd.DataFrame()
            temp['spatial_f1_macro'] = d_fold['mv_f1_macro']
            temp['spatial_oa'] = d_fold['mv_oa']
            for c in [0,1,2]:
                temp[f'spatial_sens_class_{c}'] = d_fold[f'mv_sens_class_{c}']
                temp[f'spatial_spec_class_{c}'] = d_fold[f'mv_spec_class_{c}']
            temp['model'] = ENS_MV
            temp['Category'] = 'Ensemble'
            dfs.append(temp)

    # 3. Advanced Ensembles (Soft, Thresh, Stacking)
    stack_dirs = sorted(list(OUTPUTS_DIR.glob("stacking_v2_*")))
    if stack_dirs:
        path = stack_dirs[-1] / "stacking_v2_metrics.csv"
        if path.exists():
            d = pd.read_csv(path)
            d_fold = d.groupby('fold').mean(numeric_only=True).reset_index()

            # Helper per estrarre dati
            def extract_variant(prefix, name):
                t = pd.DataFrame()
                t['spatial_f1_macro'] = d_fold[f'{prefix}_f1_macro']
                t['spatial_oa'] = d_fold.get(f'{prefix}_oa', 0.0)
                for c in [0,1,2]:
                    t[f'spatial_sens_class_{c}'] = d_fold.get(f'{prefix}_sens_class_{c}', 0)
                    t[f'spatial_spec_class_{c}'] = d_fold.get(f'{prefix}_spec_class_{c}', 0)
                t['model'] = name
                t['Category'] = 'Ensemble'
                return t

            dfs.append(extract_variant('soft', ENS_SOFT))
            dfs.append(extract_variant('thresh', ENS_THRESH))
            dfs.append(extract_variant('stack_v2', PROPOSED))

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def plot_final_comparison(df, save_path):
    print("Generazione grafico finale...")
    metrics_map = {
        'spatial_sens_class_1': 'Sens (Tumor)',
        'spatial_spec_class_1': 'Spec (Tumor)',
        'spatial_f1_macro': 'Macro F1',
        'spatial_sens_class_2': 'Sens (Vessels)'
    }
    cols = ['model', 'Category'] + list(metrics_map.keys())
    cols = [c for c in cols if c in df.columns]

    df_melt = df[cols].melt(id_vars=['model', 'Category'], var_name='Metric', value_name='Score')
    df_melt['Metric'] = df_melt['Metric'].map(metrics_map)

    palette = {m: COLOR_SINGLE for m in MODEL_ORDER}
    palette[ENS_MV] = COLOR_BASELINE
    palette[ENS_SOFT] = COLOR_SOFT
    palette[ENS_THRESH] = COLOR_THRESH
    palette[PROPOSED] = COLOR_PROPOSED

    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid", font_scale=1.2)

    g = sns.catplot(
        data=df_melt, kind="bar", x="Metric", y="Score", hue="model",
        hue_order=[m for m in MODEL_ORDER if m in df['model'].unique()],
        palette=palette, height=6, aspect=2.5, errorbar='sd',
        capsize=0.04, edgecolor='white', linewidth=0.6, alpha=0.95
    )
    g.despine(left=True)
    g.set_axis_labels("", "Score (Mean ± SD)")
    g.legend.set_title("Modello / Strategia")
    g.set(ylim=(0, 1.05))

    # Evidenzia area critica
    ax = g.ax
    # Sfondo grigio leggero sotto le metriche tumorali
    ax.axvspan(0.5, 1.5, color='gray', alpha=0.1)
    ax.text(1, 1.02, "AREA CRITICA (TUMORE)", ha='center', fontsize=10, fontweight='bold', color='#555')

    g.fig.suptitle("Analisi Dettagliata: Modelli Singoli vs Strategie Ensemble", fontsize=20, y=1.05)
    g.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafico salvato in: {save_path}")

def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    df = load_data()
    if not df.empty:
        plot_final_comparison(df, FIGURES_DIR / "comparison_metrics_ensemble_final.png")

if __name__ == "__main__":
    main()
