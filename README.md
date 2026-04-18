# Hyperspectral Imaging for Intraoperative Brain Tumor Detection
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository covers processing, training, and evaluation of Machine Learning models applied to *in-vivo* Hyperspectral Imaging (HSI) for neurosurgery. 

It originally serves as a modular re-implementation of the benchmark paper:
> *"Hyperspectral imaging benchmark based on machine learning for intraoperative brain tumour detection"* (León et al., IEEE Access).

**Beyond the baseline**, this project introduces the core contributions of my Bachelor's degree Thesis, focusing on **Risk-Aware Decision Support Systems** via advanced Ensemble architectures and Predictive Uncertainty Decomposition.

---

## Key Contributions & Features

1. **Patient-Level Cross-Validation:** A custom robust 5-fold CV split ensuring zero inter-patient data leakage.
2. **Baseline Replication:** Implementation of the full HSI preprocessing pipeline and benchmark classifiers (SVM, RF, 1D-DNN, KNN, EBEAE).
3. **Advanced Ensemble Strategies:**
   * **Weighted Soft Voting:** Probabilistic fusion maximizing global Macro F1-Score.
   * **Hierarchical Thresholding:** A safety-first rule-based approach prioritizing blood vessel preservation.
   * **Clinical Stacking V2:** A meta-learning Random Forest approach heavily penalizing False Negatives on tumor tissues.
4. **Epistemic Uncertainty Analysis:** Decomposition of predictive entropy into Aleatoric (data noise) and Epistemic (model disagreement) components.
5. **Risk-Rejection Curves:** Operational simulation demonstrating how discarding high-uncertainty voxels drastically boosts oncological sensitivity.

---

## Project Structure

```text
├── data/                  # Ignored: Place InVivoBench raw datasets here
├── models/                # ML/DL Architectures (DNN, SVM, RF, KNN, Unmixing)
├── scripts/               # CLI Executables (Train, Ensembles, Plotting)
├── utils/                 # Core logic: loading, preprocessing, spatial filtering
├── figures/               # Generated output plots and uncertainty maps
└── requirements.txt       # Minimal Python dependencies
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/HSI-benchmark-eval.git](https://github.com/YourUsername/HSI-benchmark-eval.git)
   cd HSI-benchmark-eval
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Dataset Preparation:** Download the HELICoiD/InVivoBench dataset and place the raw `.hdr` and image files inside `data/hsi_dataset/`.

---

## Usage

### 1. Preprocessing
Run the standardized preprocessing pipeline (radiometric calibration, smoothing, band removal, downsampling, normalization):
```bash
python -m scripts.preprocess
```

### 2. Training Base Models
Train a specific baseline model (e.g., Random Forest) using patient-level CV:
```bash
python -m scripts.train --model rf
```
Available models: `dnn`, `svm-l`, `svm-rbf`, `knn-e`, `knn-c`, `rf`, `ebeae`.

### 3. Advanced Ensembles & Uncertainty
Generate predictions using the Clinical Stacking V2 and Soft Voting architectures:
```bash
python -m scripts.run_stacking_ensemble_v2
```

Extract the Aleatoric and Epistemic uncertainty maps:
```bash
python -m scripts.generate_entropy_maps
```

### 4. Evaluation & Plotting
Generate the thesis charts (Risk-Rejection curves and baseline comparisons):
```bash
python -m scripts.plot_final_thesis_results
python -m scripts.plot_hybrid_risk_rejection
```

---
**Author:** Alessandro Botta
