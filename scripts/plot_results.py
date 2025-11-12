# scripts/plot_results.py
"""
Generates the results charts (Fig. 6a, 6b)
based on the 'metrics_summary.csv' files in the output folders.

Logic:
1. Scan 'outputs/' for all known models.
2. For each model, find the MOST RECENT run folder (based on timestamp).
3. Load the summary data (the 5 fold rows + 'mean'/'std').
4. Generate the F1 boxplot (Fig. 6a) from the 5-fold data.
5. Generate the bar plot (Fig. 6b) from the 'mean' and 'std' data.
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
    if "ebeae" in MODELS_TO_PLOT and "ebeae" not in latest_runs:
        print("[Plotter] ⚠ 'ebeae' data not found. Injecting hardcoded log data for Fig. 6a...")

        # Data manually extracted from log: train_ebeae_20251111_030806.log
        # (These are the per-fold averages calculated previously)
        hardcoded_ebeae_data = {
            'fold': ['1', '2', '3', '4', '5'], # Index must be 'fold'
            'spectral_f1_macro': [0.1323, 0.2375, 0.3058, 0.2670, 0.4064],
            'spatial_f1_macro': [0.1203, 0.2289, 0.3078, 0.2709, 0.3963],
            'mv_f1_macro': [0.1203, 0.1970, 0.2932, 0.2106, 0.3303],
            'model': ['EBEAE', 'EBEAE', 'EBEAE', 'EBEAE', 'EBEAE']
        }

        # Set 'fold' as the index to align with loaded data
        df_ebeae = pd.DataFrame(hardcoded_ebeae_data).set_index('fold')

        # Add the hardcoded data to the list
        all_data.append(df_ebeae)
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

    df_f1 = df_folds[['model'] + cols_to_plot]

    # 3. "Melt" the DataFrame for Seaborn
    df_melted = df_f1.melt(
        id_vars=['model'],
        value_vars=cols_to_plot,
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
        palette="muted" # Palette similar to paper
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
    Generates the bar plot for OA, Sensitivity, Specificity (like Fig. 6b).
    Uses only the 'Spatial/Spectral' pipeline.
    """
    print("[Plotter] Generating Fig. 6b (Metrics Bar Plot)...")

    # 1. Filter 'mean' and 'std' data
    if 'mean' not in summary_data.index or 'std' not in summary_data.index:
        print("[Plotter] ❌ 'mean' or 'std' rows not found. Cannot generate Fig. 6b.")
        return

    # Use [[]] to select as DataFrame and .copy() to avoid warnings
    df_mean_full = summary_data.loc[['mean']].copy()
    df_std_full = summary_data.loc[['std']].copy()

    # 2. Define and map metrics
    metrics_map = {
        'spatial_oa': 'OA',
        'spatial_sens_class_0': 'Sens (NT)',
        'spatial_sens_class_1': 'Sens (TT)',
        'spatial_sens_class_2': 'Sens (BV)',
        # 'spatial_sens_class_3': 'Sens (BG)', # Add if you evaluate BG
        'spatial_spec_class_0': 'Spec (NT)',
        'spatial_spec_class_1': 'Spec (TT)',
        'spatial_spec_class_2': 'Spec (BV)',
        # 'spatial_spec_class_3': 'Spec (BG)', # Add if you evaluate BG
    }

    # Check which columns are actually available
    cols_to_plot = [col for col in metrics_map.keys() if col in df_mean_full.columns]

    if not cols_to_plot:
        print("[Plotter] ❌ No 'spatial' (oa/sens/spec) metrics found. Cannot generate Fig. 6b.")
        print("   (Did you run train.py *after* modifying it to save all metrics?)")
        return

    # Select only the columns we need
    df_mean = df_mean_full[['model'] + cols_to_plot]
    df_std = df_std_full[['model'] + cols_to_plot]

    # 3. Melt for Seaborn
    df_mean_melted = df_mean.melt(id_vars='model', var_name='Metric', value_name='Mean')
    df_std_melted = df_std.melt(id_vars='model', var_name='Metric', value_name='Std')

    # Merge mean and std
    df_plot = pd.merge(df_mean_melted, df_std_melted, on=['model', 'Metric'])
    df_plot['Metric'] = df_plot['Metric'].map(metrics_map)

    # Filter out any metrics not in our map (e.g., BG if we skipped it)
    df_plot = df_plot.dropna(subset=['Metric'])

    # 4. Create the plot (using catplot for facets)
    g = sns.catplot(
        data=df_plot,
        x='model',
        y='Mean',
        col='Metric',
        kind='bar',
        order=MODEL_ORDER,
        palette='viridis',
        col_wrap=4, # 4 charts per row
        height=4,
        aspect=1.2,
        sharey=False # Allow different y-axes if needed
    )

    g.fig.suptitle("Fig. 6b (Replica): 'Spatial/Spectral' Metrics (5-Fold Mean)", y=1.03, fontsize=16)
    g.set_axis_labels("Classifier", "Mean Value")
    g.set_xticklabels(rotation=45)

    # Set Y-lim and grid for all axes
    for ax in g.axes.flatten():
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add error bars
    metric_order = [m for m in metrics_map.values() if m in df_plot['Metric'].unique()]

    for i, metric in enumerate(metric_order):
        ax = g.axes.flatten()[i]
        sub_df = df_plot[df_plot['Metric'] == metric]

        # Sort the sub-dataframe to match the plot's x-axis order
        sub_df = sub_df.set_index('model').reindex(MODEL_ORDER).reset_index()

        ax.errorbar(
            x=sub_df['model'],
            y=sub_df['Mean'],
            yerr=sub_df['Std'],
            fmt='none',
            capsize=5,
            color='black'
        )

    # 5. Save the file
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.close()
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
