HSI Benchmark Evaluation (Python)
=================================

This repository is a Python re-implementation of the paper:

"Hyperspectral imaging benchmark based on machine learning for intraoperative brain tumour detection"
IEEE Access, HIRIS-Lab

This project provides a modular Python version to replicate and extend the experiments.

----------------------------------------------------------------------
Project Structure
----------------------------------------------------------------------

configs/             -> Experiment configuration files (YAML)
data/                -> Input datasets (ignored in git)
   raw/              -> Original raw datasets
   processed/        -> Preprocessed data (numpy/tensors)
   external/         -> External or pre-trained models
experiments/         -> Training and evaluation experiments
   notebooks/        -> Jupyter notebooks for EDA & visualization
models/              -> Neural network architectures
outputs/             -> Logs, checkpoints, figures (ignored in git)
scripts/             -> CLI scripts (train, evaluate, preprocess)
utils/               -> Helper functions and preprocessing
main.py              -> Optional main entrypoint
requirements.txt     -> Python dependencies
environment.yml      -> Conda environment (optional)
README.md            -> Project documentation

----------------------------------------------------------------------
Setup
----------------------------------------------------------------------

1. Clone the repository:
   git clone <this-repo-url>
   cd HSI-benchmark-eval

2. Create and activate a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate   # on Linux/Mac
   .venv\Scripts\activate      # on Windows PowerShell

3. Install dependencies with pip:
   pip install --upgrade pip
   pip install -r requirements.txt

4. Place datasets into the data/raw/ folder.
   (Datasets are not included in the repository due to size.)


----------------------------------------------------------------------
Usage
----------------------------------------------------------------------

Preprocess dataset:
   python scripts/preprocess.py --config configs/default.yaml

Train a model:
   python scripts/train.py --config configs/exp1_brain.yaml

Evaluate a model:
   python scripts/evaluate.py --checkpoint outputs/checkpoints/exp1.pth

