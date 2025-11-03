# models/dnn_runner.py
"""
DNNRunner
---------
Implements a simple 1D fully connected neural network classifier
for hyperspectral pixel-level classification.

Paper-aligned defaults:
    - Two hidden layers with BatchNorm + ReLU
    - SGD + momentum 0.9
    - learning_rate = 0.1
    - epochs = 300
    - Class mapping fixed to {0:NT, 1:TT, 2:BV, 3:BG}
    - Probability channel order [NT, TT, BV, BG]
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from . import BaseRunner

class DNN1D(nn.Module):
    def __init__(self, input_dim=128, num_classes=4, hidden_dims=(256, 256)):
        super().__init__()
        assert len(hidden_dims) == 2, "This paper-aligned DNN expects exactly two hidden layers."

        h1, h2 = hidden_dims

        # Hidden layer 1: Linear -> ReLU (no BatchNorm here)
        layer1 = [
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
        ]

        # Hidden layer 2: Linear -> BatchNorm -> ReLU
        layer2 = [
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
        ]

        # Output layer
        out = [nn.Linear(h2, num_classes)]

        self.net = nn.Sequential(*(layer1 + layer2 + out))

    def forward(self, x):
        return self.net(x)


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
                 hidden_dims=(256, 256),
                 learning_rate=0.1,
                 batch_size=512,
                 epochs=300,
                 momentum=0.9,
                 weight_decay=0.0,
                 device="cuda"):
        self.name = "dnn"
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Device setup
        use_cuda = (device == "cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Model (will be rebuilt on fit if input_dim mismatches)
        self.net = DNN1D(input_dim=input_dim,
                        num_classes=num_classes,
                        hidden_dims=hidden_dims).to(self.device)
        self.softmax = nn.Softmax(dim=1)

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _maybe_rebuild(self, input_dim):
        """Rebuild network if input dimension does not match."""
        if input_dim != self.input_dim:
            self.input_dim = input_dim
            self.net = DNN1D(input_dim=input_dim,
                             num_classes=self.num_classes,
                             hidden_dims=self.hidden_dims).to(self.device)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def fit(self, X, y):
        """Train the DNN on flattened spectral data (X: N×D, y: N in {0..3})."""
        if X.size == 0:
            print("[DNNRunner] ⚠ Empty training set, skipping training.")
            return

        # Ensure input dim matches data
        self._maybe_rebuild(X.shape[1])

        # Convert to tensors
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Inverse-frequency class weights (normalized)
        cls, cnt = np.unique(y, return_counts=True)
        weights = np.zeros(self.num_classes, dtype=np.float32)
        weights[cls] = 1.0 / (cnt + 1e-8)
        weights = torch.tensor(weights / (weights.sum() + 1e-8) * len(weights), dtype=torch.float32).to(self.device)

        # Loss and optimizer (paper: SGD + momentum, high LR)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=self.learning_rate,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

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

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"[DNNRunner] Epoch {epoch+1}/{self.epochs} | "
                    f"LR: {self.learning_rate:.4f} | Loss: {total_loss/len(dl):.4f}")

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    def _predict_proba(self, X):
        """Return softmax probabilities for input spectra; channels [NT, TT, BV, BG]."""
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32).to(self.device))
            return self.softmax(logits).cpu().numpy()

    def predict_full(self, cube):
        """
        Predict class map and probability map for a full HSI cube.
        Returns:
            class_map: (H, W) in {0:NT,1:TT,2:BV,3:BG}
            prob_all:  (H, W, num_classes) with channel order [NT, TT, BV, BG]
        """
        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T
        # Ensure network has correct input dim if used standalone
        self._maybe_rebuild(flat.shape[1])

        proba = self._predict_proba(flat)
        class_map = np.argmax(proba, axis=1).reshape(H, W)
        prob_all = proba.reshape(H, W, -1)
        return class_map, prob_all
