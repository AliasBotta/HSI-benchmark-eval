# scripts/plot_results.py
"""
Genera i grafici dei risultati (Fig. 6a, 6b)
basandosi sui file 'metrics_summary.csv' nelle cartelle di output.

Logica:
1. Scansiona 'outputs/' per tutti i modelli noti.
2. Per ogni modello, trova la cartella della run PIÙ RECENTE (basata sul timestamp).
3. Carica i dati di sommario (le 5 righe dei fold + 'mean'/'std').
4. Genera il boxplot F1 (Fig. 6a) dai dati dei 5 fold.
5. Genera il bar plot (Fig. 6b) dai dati 'mean' e 'std'.
6. Salva i grafici in 'outputs/'.

Assicurati di installare seaborn: pip install seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Directory di base (dove si trova questo script)
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures" # Salva i grafici in una cartella dedicata

# Modelli da cercare (come definiti nel paper)
MODELS_TO_PLOT = [
    "ebeae", "nebeae", "rf", "knn-e", 
    "knn-c", "svm-l", "svm-rbf", "dnn"
]
# Ordine corretto per i grafici
MODEL_ORDER = [
    "EBEAE", "NEBEAE", "RF", "KNN-E", 
    "KNN-C", "SVM-L", "SVM-RBF", "DNN"
]

def find_latest_run_folders(outputs_dir: Path, model_names: list) -> dict:
    """
    Trova la cartella della run più recente per ogni modello.
    """
    latest_runs = {}
    for model_name in model_names:
        run_folders = sorted(list(outputs_dir.glob(f"{model_name}_*")))
        
        if not run_folders:
            print(f"[Plotter] ⚠ Nessuna run trovata per: {model_name}")
            continue
            
        latest_run = run_folders[-1] # L'ultima è la più recente
        
        # Controlla che il file di summary esista
        if not (latest_run / "metrics_summary.csv").exists():
            print(f"[Plotter] ⚠ Run trovata per {model_name} ma manca 'metrics_summary.csv': {latest_run.name}")
            continue
            
        print(f"[Plotter] Trovata run valida per {model_name}: {latest_run.name}")
        latest_runs[model_name] = latest_run
        
    return latest_runs

def load_summary_data(latest_runs: dict) -> pd.DataFrame:
    """
    Carica tutti i file 'metrics_summary.csv' in un unico DataFrame.
    """
    all_data = []
    for model_name, folder_path in latest_runs.items():
        try:
            df = pd.read_csv(folder_path / "metrics_summary.csv", index_col=0)
            df['model'] = model_name.upper() # Es. 'rf' -> 'RF'
            all_data.append(df)
        except Exception as e:
            print(f"[Plotter] ❌ Errore nel caricamento di {folder_path / 'metrics_summary.csv'}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data)

def plot_fig6a_f1_boxplot(summary_data: pd.DataFrame, save_path: Path):
    """
    Genera il boxplot del Macro F1-Score (come Fig. 6a).
    """
    print("[Plotter] Generazione Fig. 6a (Boxplot Macro F1)...")
    
    # 1. Filtra solo i dati dei 5 fold (escludi 'mean' e 'std')
    fold_indices = [str(i) for i in range(1, 6)] + [i for i in range(1, 6)]
    df_folds = summary_data[summary_data.index.isin(fold_indices)].copy()
    
    if df_folds.empty:
        print("[Plotter] ❌ Dati dei fold non trovati. Impossibile generare Fig. 6a.")
        return

    # 2. Seleziona e rinomina le colonne F1
    f1_cols = {
        "spectral_f1_macro": "Spectral",
        "spatial_f1_macro": "Spatial/Spectral",
        "mv_f1_macro": "Majority Voting"
    }
    df_f1 = df_folds[['model'] + list(f1_cols.keys())]
    
    # 3. "Melt" (sciogli) il DataFrame per Seaborn
    df_melted = df_f1.melt(
        id_vars=['model'],
        value_vars=list(f1_cols.keys()),
        var_name='Pipeline',
        value_name='Macro F1-Score'
    )
    df_melted['Pipeline'] = df_melted['Pipeline'].map(f1_cols)
    
    # 4. Crea il plot
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=df_melted,
        x='model',
        y='Macro F1-Score',
        hue='Pipeline',
        order=MODEL_ORDER, # Applica l'ordine corretto
        palette="muted" # Palette simile a quella del paper
    )
    
    plt.title("Fig. 6a (Replica): Macro F1-Score (Test Set, 5 Folds)", fontsize=16)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.xlabel("Classifier", fontsize=12)
    plt.ylim(0.3, 1.0) # Allineato al paper
    plt.legend(title="Pipeline")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 5. Salva il file
    plt.savefig(save_path)
    plt.close()
    print(f"[Plotter] ✅ Grafico salvato in: {save_path}")

def plot_fig6b_bars(summary_data: pd.DataFrame, save_path: Path):
    """
    Genera il bar plot di OA, Sensitivity, Specificity (come Fig. 6b).
    Usa solo la pipeline 'Spatial/Spectral'.
    """
    print("[Plotter] Generazione Fig. 6b (Bar Plot Metriche)...")
    
    # 1. Filtra i dati 'mean' e 'std'
    if 'mean' not in summary_data.index or 'std' not in summary_data.index:
        print("[Plotter] ❌ Dati 'mean' o 'std' non trovati. Impossibile generare Fig. 6b.")
        return
        
    df_mean = summary_data.loc['mean']
    df_std = summary_data.loc['std']
    df_mean['model'] = summary_data.loc['mean', 'model']
    df_std['model'] = summary_data.loc['std', 'model']

    # 2. Definisci le metriche e rinominale
    # NOTA: Il tuo script al momento calcola solo NT, TT, BV (classi 0, 1, 2)
    metrics_map = {
        'spatial_oa': 'OA',
        'spatial_sens_class_0': 'Sens (NT)',
        'spatial_sens_class_1': 'Sens (TT)',
        'spatial_sens_class_2': 'Sens (BV)',
        'spatial_spec_class_0': 'Spec (NT)',
        'spatial_spec_class_1': 'Spec (TT)',
        'spatial_spec_class_2': 'Spec (BV)',
    }
    
    # Controlla se le colonne esistono (in caso di run parziali)
    cols_to_plot = [col for col in metrics_map.keys() if col in df_mean.columns]
    if not cols_to_plot:
        print("[Plotter] ❌ Nessuna metrica 'spatial' (oa/sens/spec) trovata. Impossibile generare Fig. 6b.")
        return

    df_mean_plot = df_mean[['model'] + cols_to_plot]
    df_std_plot = df_std[['model'] + cols_to_plot]

    # 3. Melt per Seaborn
    df_mean_melted = df_mean_plot.melt(id_vars='model', var_name='Metric', value_name='Mean')
    df_std_melted = df_std_plot.melt(id_vars='model', var_name='Metric', value_name='Std')
    
    # Unisci mean e std
    df_plot = pd.merge(df_mean_melted, df_std_melted, on=['model', 'Metric'])
    df_plot['Metric'] = df_plot['Metric'].map(metrics_map)
    
    # 4. Crea il plot (usando catplot per griglie)
    g = sns.catplot(
        data=df_plot,
        x='model',
        y='Mean',
        col='Metric',
        kind='bar',
        order=MODEL_ORDER,
        palette='viridis',
        col_wrap=4, # 4 grafici per riga
        height=4,
        aspect=1.2
    )
    
    g.fig.suptitle("Fig. 6b (Replica): Metriche 'Spatial/Spectral' (Media 5 Folds)", y=1.03, fontsize=16)
    g.set_axis_labels("Classifier", "Mean Value")
    g.set_xticklabels(rotation=45)
    g.set(ylim=(0, 1.1))

    # Aggiungi le barre d'errore (catplot non lo fa facilmente con 'yerr')
    for ax, (_, sub_df) in zip(g.axes.flatten(), df_plot.groupby('Metric')):
        # Ordina il sub_df per matchare l'ordine del grafico
        sub_df = sub_df.set_index('model').loc[MODEL_ORDER].reset_index()
        
        ax.errorbar(
            x=sub_df['model'],
            y=sub_df['Mean'],
            yerr=sub_df['Std'],
            fmt='none',
            capsize=5,
            color='black'
        )

    # 5. Salva il file
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close()
    print(f"[Plotter] ✅ Grafico salvato in: {save_path}")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    
    # 1. Trova le run più recenti
    latest_runs = find_latest_run_folders(OUTPUTS_DIR, MODELS_TO_PLOT)
    
    if not latest_runs:
        print("[Plotter] ❌ Nessuna run valida trovata in 'outputs/'. Interruzione.")
        sys.exit(1)
        
    # 2. Carica i dati
    summary_data = load_summary_data(latest_runs)
    
    if summary_data.empty:
        print("[Plotter] ❌ Dati di summary vuoti. Interruzione.")
        sys.exit(1)
        
    # 3. Genera Grafico 6a (Boxplot)
    plot_fig6a_f1_boxplot(
        summary_data,
        FIGURES_DIR / "fig6a_f1_boxplot.png"
    )
    
    # 4. Genera Grafico 6b (Barre)
    plot_fig6b_bars(
        summary_data,
        FIGURES_DIR / "fig6b_metrics_barplot.png"
    )

if __name__ == "__main__":
    main()
