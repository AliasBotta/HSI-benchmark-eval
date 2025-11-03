# models/dnn_runner.py
"""
DNNRunner
---------
Implements a simple 1D fully connected neural network classifier
for hyperspectral pixel-level classification.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from . import BaseRunner
from .dnn_1d import DNN1D


class DNNRunner(BaseRunner):
    """
    Deep Neural Network runner (1D fully connected model).

    Methods:
        - fit(X, y)
        - predict_full(cube)
    """

    def __init__(self,
                 input_dim=128,
                 num_classes=4,
                 hidden_dims=(256, 128),
                 learning_rate=1e-3,
                 batch_size=256,
                 epochs=15,
                 momentum=0.9,
                 device="cuda"):
        self.name = "dnn"
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum

        # Device setup
        use_cuda = (device == "cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Model and loss
        self.net = DNN1D(input_dim=input_dim,
                         num_classes=num_classes,
                         hidden_dims=hidden_dims).to(self.device)
        self.softmax = nn.Softmax(dim=1)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def fit(self, X, y):
        """Train the DNN on flattened spectral data."""
        if X.size == 0:
            print("[DNNRunner] ⚠ Empty training set, skipping training.")
            return

        # Convert to tensors
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        # Compute inverse-frequency class weights
        cls, cnt = np.unique(y, return_counts=True)
        weights = np.zeros(self.num_classes, dtype=np.float32)
        weights[cls] = 1.0 / (cnt + 1e-8)
        weights = torch.tensor(weights / weights.sum() * len(weights), dtype=torch.float32).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=self.learning_rate,
                                    momentum=self.momentum)

        # Training loop
        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[DNNRunner] Epoch {epoch+1}/{self.epochs} | Loss: {total_loss/len(dl):.4f}")

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    def _predict_proba(self, X):
        """Return softmax probabilities for input spectra."""
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32).to(self.device))
            return self.softmax(logits).cpu().numpy()

    def predict_full(self, cube):
        """
        Predict class map and probability map for a full HSI cube.
        Returns:
            class_map: (H, W)
            prob_all:  (H, W, num_classes)
        """
        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T
        proba = self._predict_proba(flat)
        class_map = np.argmax(proba, axis=1).reshape(H, W)
        prob_all = proba.reshape(H, W, -1)
        return class_map, prob_all
