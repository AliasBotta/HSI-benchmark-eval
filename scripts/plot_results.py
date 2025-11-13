"""
Generates the results charts (Fig. 6a, 6b)
based on the 'metrics_summary.csv' files in the output folders.

Logic:
1. Scan 'outputs/' for all known models.
2. For each model, find the MOST RECENT run folder (based on timestamp).
3. Load the summary data (the 5 fold rows + 'mean'/'std').
4. Generate the F1 boxplot (Fig. 6a) from the 5-fold data.
5. Generate the grouped bar plot (Fig. 6b) from the 5-fold data.
6. Save the charts to the 'figures/' directory.

Ensure seaborn is installed: pip install seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np 

BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures" 

MODELS_TO_PLOT = [
    "ebeae", "nebeae", "rf", "knn-e",
    "knn-c", "svm-l", "svm-rbf", "dnn"
]
MODEL_ORDER = [
    "EBEAE", "NEBEAE", "RF", "KNN-E",
    "KNN-C", "SVM-L", "SVM-RBF", "DNN"
]

def find_latest_run_folders(outputs_dir: Path, model_names: list) -> dict:
    """
    Finds the most recent run folder for each model.
    """
    latest_runs = {}
    for model_name in model_names:
        run_folders = sorted(list(outputs_dir.glob(f"{model_name}_*")))

        if not run_folders:
            print(f"[Plotter] ⚠ No run found for: {model_name}")
            continue

        latest_run = run_folders[-1] 

        if not (latest_run / "metrics_summary.csv").exists():
            print(f"[Plotter] ⚠ Run found for {model_name} but missing 'metrics_summary.csv': {latest_run.name}")
            continue

        print(f"[Plotter] Found valid run for {model_name}: {latest_run.name}")
        latest_runs[model_name] = latest_run

    return latest_runs

def load_summary_data(latest_runs: dict) -> pd.DataFrame:
    """
    Loads all 'metrics_summary.csv' files into a single DataFrame.
    INCLUDES: Hardcoded data for 'ebeae' (Fig 6a) from log 20251111_030806.log
    """
    all_data = []
    for model_name, folder_path in latest_runs.items():
        try:
            df = pd.read_csv(folder_path / "metrics_summary.csv", index_col=0)
            df['model'] = model_name.upper() 
            all_data.append(df)
        except Exception as e:
            print(f"[Plotter] ❌ Error loading {folder_path / 'metrics_summary.csv'}: {e}")


    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data)

def plot_fig6a_f1_boxplot(summary_data: pd.DataFrame, save_path: Path):
    """
    Generates the Macro F1-Score boxplot (like Fig. 6a).
    """
    print("[Plotter] Generating Fig. 6a (Macro F1 Boxplot)...")

    fold_indices = [str(i) for i in range(1, 6)] + [i for i in range(1, 6)]
    df_folds = summary_data[summary_data.index.isin(fold_indices)].copy()

    if df_folds.empty:
        print("[Plotter] ❌ Fold data not found. Cannot generate Fig. 6a.")
        return

    f1_cols_map = {
        "spectral_f1_macro": "Spectral",
        "spatial_f1_macro": "Spatial/Spectral",
        "mv_f1_macro": "Majority Voting",
        "spectral_f1": "Spectral", 
        "spatial_f1": "Spatial/Spectral", 
        "mv_f1": "Majority Voting" 
    }

    cols_to_plot = [col for col in f1_cols_map if col in df_folds.columns]

    if not cols_to_plot:
        print("[Plotter] ❌ No F1-Score columns found (e.g., 'spatial_f1_macro'). Cannot generate Fig. 6a.")
        return

    df_f1 = df_folds[['model'] + list(set(cols_to_plot))] 

    df_melted = df_f1.melt(
        id_vars=['model'],
        value_vars=list(set(cols_to_plot)),
        var_name='Pipeline',
        value_name='Macro F1-Score'
    )
    df_melted['Pipeline'] = df_melted['Pipeline'].map(f1_cols_map)

    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=df_melted,
        x='model',
        y='Macro F1-Score',
        hue='Pipeline',
        order=MODEL_ORDER, 
        palette="muted", 
        notch=True       
    )

    plt.title("Fig. 6a (Replica): Macro F1-Score (Test Set, 5 Folds)", fontsize=16)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.xlabel("Classifier", fontsize=12)
    plt.ylim(0.0, 1.0) 
    plt.legend(title="Pipeline")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path)
    plt.close()
    print(f"[Plotter] ✅ Figure saved to: {save_path}")

def plot_fig6b_bars(summary_data: pd.DataFrame, save_path: Path):
    """
    Generates the grouped bar plot for OA, Sensitivity, Specificity (like Fig. 6b).
    Uses only the 'Spatial/Spectral' pipeline data from the 5 folds.
    """
    print("[Plotter] Generating Fig. 6b (Grouped Metrics Bar Plot)...")

    fold_indices = [str(i) for i in range(1, 6)] + [i for i in range(1, 6)]
    df_folds = summary_data[summary_data.index.isin(fold_indices)].copy()

    if df_folds.empty:
        print("[Plotter] ❌ Fold data not found. Cannot generate Fig. 6b.")
        return

    metrics_map = {
        'spatial_oa': 'OA',
        'spatial_sens_class_0': 'Sens (NT)',
        'spatial_sens_class_1': 'Sens (TT)',
        'spatial_sens_class_2': 'Sens (BV)',
        'spatial_spec_class_0': 'Spec (NT)',
        'spatial_spec_class_1': 'Spec (TT)',
        'spatial_spec_class_2': 'Spec (BV)',
    }
    metric_order = [
        'OA', 'Sens (NT)', 'Sens (TT)', 'Sens (BV)',
        'Spec (NT)', 'Spec (TT)', 'Spec (BV)'
    ]

    cols_to_plot = [col for col in metrics_map.keys() if col in df_folds.columns]
    if not cols_to_plot:
        print("[Plotter] ❌ No 'spatial' (oa/sens/spec) metrics found. Cannot generate Fig. 6b.")
        print("    (Did you run train.py *after* modifying it to save all metrics?)")
        return

    df_metrics = df_folds[['model'] + cols_to_plot]

    df_melted = df_metrics.melt(
        id_vars=['model'],
        value_vars=cols_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    df_melted['Metric'] = df_melted['Metric'].map(metrics_map)
    df_melted = df_melted.dropna(subset=['Metric'])

    g = sns.catplot(
        data=df_melted,
        x='Metric',           
        y='Value',            
        hue='model',          
        kind='bar',
        order=metric_order,   
        hue_order=MODEL_ORDER,
        palette='deep',       
        errorbar='sd',        
        legend=True,          
        height=7,
        aspect=2.5
    )

    g.set_axis_labels("Metric", "Mean Value")
    g.set_xticklabels(rotation=0)
    g.set(ylim=(0, 1.1))

    g.ax.grid(axis='y', linestyle='--', alpha=0.7)

    sns.move_legend(
        g,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15), 
        ncol=8,
        title='Classifier',
        frameon=True
    )

    g.fig.savefig(save_path, bbox_inches='tight')
    plt.close('all') 
    print(f"[Plotter] ✅ Figure saved to: {save_path}")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    latest_runs = find_latest_run_folders(OUTPUTS_DIR, MODELS_TO_PLOT)

    if not latest_runs:
        print("[Plotter] ❌ No valid runs found in 'outputs/'. Exiting.")
        sys.exit(1)

    summary_data = load_summary_data(latest_runs)

    if summary_data.empty:
        print("[Plotter] ❌ Empty summary data. Exiting.")
        sys.exit(1)

    plot_fig6a_f1_boxplot(
        summary_data,
        FIGURES_DIR / "fig6a_f1_boxplot.png"
    )

    plot_fig6b_bars(
        summary_data,
        FIGURES_DIR / "fig6b_metrics_barplot.png"
    )

if __name__ == "__main__":
    main()
