# scripts/plot_results.py
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
import numpy as np # Added for hardcoded data

# Base directory (where this script is located)
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures" # Save charts in a dedicated folder

# Models to search for (as defined in the paper)
MODELS_TO_PLOT = [
    "ebeae", "nebeae", "rf", "knn-e",
    "knn-c", "svm-l", "svm-rbf", "dnn"
]
# Correct order for the plots
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

        latest_run = run_folders[-1] # The last one is the most recent

        # Check that the summary file exists
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
            df['model'] = model_name.upper() # e.g., 'rf' -> 'RF'
            all_data.append(df)
        except Exception as e:
            print(f"[Plotter] ❌ Error loading {folder_path / 'metrics_summary.csv'}: {e}")

    # <--- START MODIFICATION: Hardcode EBEAE if not found ---
    # Check if 'ebeae' was requested but NOT found
    # if "ebeae" in MODELS_TO_PLOT and "ebeae" not in latest_runs:
    #     print("[Plotter] ⚠ 'ebeae' data not found. Injecting hardcoded log data for Fig. 6a...")
    #
    #     # Data manually extracted from log: train_ebeae_20251111_030806.log
    #     # (These are the per-fold averages calculated previously)
    #     hardcoded_ebeae_data = {
    #         'fold': ['1', '2', '3', '4', '5'], # Index must be 'fold'
    #         'spectral_f1_macro': [0.1323, 0.2375, 0.3058, 0.2670, 0.4064],
    #         'spatial_f1_macro': [0.1203, 0.2289, 0.3078, 0.2709, 0.3963],
    #         'mv_f1_macro': [0.1203, 0.1970, 0.2932, 0.2106, 0.3303],
    #         'model': ['EBEAE', 'EBEAE', 'EBEAE', 'EBEAE', 'EBEAE']
    #     }
    #
    #     # Set 'fold' as the index to align with loaded data
    #     df_ebeae = pd.DataFrame(hardcoded_ebeae_data).set_index('fold')
    #
    #     # Add the hardcoded data to the list
    #     all_data.append(df_ebeae)
    # <--- END MODIFICATION ---

    if not all_data:
        return pd.DataFrame()

    # Concatenate all DataFrames (loaded + hardcoded)
    return pd.concat(all_data)

def plot_fig6a_f1_boxplot(summary_data: pd.DataFrame, save_path: Path):
    """
    Generates the Macro F1-Score boxplot (like Fig. 6a).
    """
    print("[Plotter] Generating Fig. 6a (Macro F1 Boxplot)...")

    # 1. Filter only the 5-fold data (exclude 'mean' and 'std')
    fold_indices = [str(i) for i in range(1, 6)] + [i for i in range(1, 6)]
    df_folds = summary_data[summary_data.index.isin(fold_indices)].copy()

    if df_folds.empty:
        print("[Plotter] ❌ Fold data not found. Cannot generate Fig. 6a.")
        return

    # 2. Select and rename F1 columns
    # Use 'spectral_f1_macro' if available, fallback to 'spectral_f1'
    f1_cols_map = {
        "spectral_f1_macro": "Spectral",
        "spatial_f1_macro": "Spatial/Spectral",
        "mv_f1_macro": "Majority Voting",
        "spectral_f1": "Spectral", # Fallback for old logs
        "spatial_f1": "Spatial/Spectral", # Fallback for old logs
        "mv_f1": "Majority Voting" # Fallback for old logs
    }

    # Find which columns actually exist in the dataframe
    cols_to_plot = [col for col in f1_cols_map if col in df_folds.columns]

    if not cols_to_plot:
        print("[Plotter] ❌ No F1-Score columns found (e.g., 'spatial_f1_macro'). Cannot generate Fig. 6a.")
        return

    df_f1 = df_folds[['model'] + list(set(cols_to_plot))] # Use set() to avoid duplicates

    # 3. "Melt" the DataFrame for Seaborn
    df_melted = df_f1.melt(
        id_vars=['model'],
        value_vars=list(set(cols_to_plot)),
        var_name='Pipeline',
        value_name='Macro F1-Score'
    )
    # Map to clean names (e.g., 'spectral_f1_macro' -> 'Spectral')
    df_melted['Pipeline'] = df_melted['Pipeline'].map(f1_cols_map)

    # 4. Create the plot
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=df_melted,
        x='model',
        y='Macro F1-Score',
        hue='Pipeline',
        order=MODEL_ORDER, # Apply correct order
        palette="muted", # Palette similar to paper
        notch=True       # <--- MODIFICATION: This adds the notches
    )

    plt.title("Fig. 6a (Replica): Macro F1-Score (Test Set, 5 Folds)", fontsize=16)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.xlabel("Classifier", fontsize=12)
    plt.ylim(0.0, 1.0) # Start from 0 for better context, paper uses 0.3
    plt.legend(title="Pipeline")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. Save the file
    plt.savefig(save_path)
    plt.close()
    print(f"[Plotter] ✅ Figure saved to: {save_path}")

def plot_fig6b_bars(summary_data: pd.DataFrame, save_path: Path):
    """
    Generates the grouped bar plot for OA, Sensitivity, Specificity (like Fig. 6b).
    Uses only the 'Spatial/Spectral' pipeline data from the 5 folds.
    """
    print("[Plotter] Generating Fig. 6b (Grouped Metrics Bar Plot)...")

    # 1. Filter only the 5-fold data (exclude 'mean' and 'std')
    fold_indices = [str(i) for i in range(1, 6)] + [i for i in range(1, 6)]
    df_folds = summary_data[summary_data.index.isin(fold_indices)].copy()

    if df_folds.empty:
        print("[Plotter] ❌ Fold data not found. Cannot generate Fig. 6b.")
        return

    # 2. Define metrics to plot
    # We only use the 'spatial' pipeline results
    metrics_map = {
        'spatial_oa': 'OA',
        'spatial_sens_class_0': 'Sens (NT)',
        'spatial_sens_class_1': 'Sens (TT)',
        'spatial_sens_class_2': 'Sens (BV)',
        'spatial_spec_class_0': 'Spec (NT)',
        'spatial_spec_class_1': 'Spec (TT)',
        'spatial_spec_class_2': 'Spec (BV)',
    }
    # Define the order for the X-axis
    metric_order = [
        'OA', 'Sens (NT)', 'Sens (TT)', 'Sens (BV)',
        'Spec (NT)', 'Spec (TT)', 'Spec (BV)'
    ]

    # 3. Check which columns are available
    cols_to_plot = [col for col in metrics_map.keys() if col in df_folds.columns]
    if not cols_to_plot:
        print("[Plotter] ❌ No 'spatial' (oa/sens/spec) metrics found. Cannot generate Fig. 6b.")
        print("    (Did you run train.py *after* modifying it to save all metrics?)")
        return

    df_metrics = df_folds[['model'] + cols_to_plot]

    # 4. "Melt" the DataFrame for Seaborn
    df_melted = df_metrics.melt(
        id_vars=['model'],
        value_vars=cols_to_plot,
        var_name='Metric',
        value_name='Value'
    )
    df_melted['Metric'] = df_melted['Metric'].map(metrics_map)
    df_melted = df_melted.dropna(subset=['Metric'])

    # 5. Create the plot
    # Use catplot (kind='bar') as it handles grouping and error bars ('sd') correctly
    g = sns.catplot(
        data=df_melted,
        x='Metric',           # Metrics on X-axis
        y='Value',            # Value on Y-axis
        hue='model',          # Model as color
        kind='bar',
        order=metric_order,   # Use defined metric order
        hue_order=MODEL_ORDER,# Use defined model order
        palette='deep',       # Palette similar to paper
        errorbar='sd',        # Use 'sd' for standard deviation error bars
        legend=True,          # Let Seaborn create the legend
        height=7,
        aspect=2.5
    )

    g.set_axis_labels("Metric", "Mean Value")
    g.set_xticklabels(rotation=0)
    g.set(ylim=(0, 1.1))

    # Add grid to the axes
    g.ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Use sns.move_legend() to move the legend
    sns.move_legend(
        g,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15), # Position it above the title
        ncol=8,
        title='Classifier',
        frameon=True
    )

    # 6. Save the file
    # Use g.fig.savefig() to ensure the moved legend is included
    g.fig.savefig(save_path, bbox_inches='tight')
    plt.close('all') # Close all figures
    print(f"[Plotter] ✅ Figure saved to: {save_path}")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    # 1. Find the most recent runs
    latest_runs = find_latest_run_folders(OUTPUTS_DIR, MODELS_TO_PLOT)

    if not latest_runs:
        print("[Plotter] ❌ No valid runs found in 'outputs/'. Exiting.")
        sys.exit(1)

    # 2. Load the data
    summary_data = load_summary_data(latest_runs)

    if summary_data.empty:
        print("[Plotter] ❌ Empty summary data. Exiting.")
        sys.exit(1)

    # 3. Generate Chart 6a (Boxplot)
    plot_fig6a_f1_boxplot(
        summary_data,
        FIGURES_DIR / "fig6a_f1_boxplot.png"
    )

    # 4. Generate Chart 6b (Bar Plot)
    plot_fig6b_bars(
        summary_data,
        FIGURES_DIR / "fig6b_metrics_barplot.png"
    )

if __name__ == "__main__":
    main()
