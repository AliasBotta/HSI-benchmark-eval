# /home/ale/repos/HSI-benchmark-eval/models/dnn_runner.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from . import BaseRunner
import copy
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import MultiStepLR

class DNN1D(nn.Module):
    def __init__(self, input_dim=128, num_classes=4, hidden_dims=(256, 256), dropout_rate=0.5): # <-- ADDED DROPOUT
        super().__init__()
        assert len(hidden_dims) == 2, "This paper-aligned DNN expects exactly two hidden layers."

        h1, h2 = hidden_dims

        # Layer 1: Linear -> ReLU -> Dropout
        layer1 = [
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) # <-- ADDED
        ]

        # Layer 2: Linear -> ReLU -> Dropout
        layer2 = [
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) # <-- ADDED
        ]

        # Batch Norm *follows* the two hidden layers (as per paper text)
        bn_layer = [
            nn.BatchNorm1d(h2)
        ]

        # Output layer
        out = [nn.Linear(h2, num_classes)]

        self.net = nn.Sequential(*(layer1 + layer2 + bn_layer + out))

    def forward(self, x):
        return self.net(x)


class DNNRunner(BaseRunner):
    """
    Deep Neural Network runner (1D fully connected model).
    Implements coarse search for hidden layer size (L) during fit.
    """

    def __init__(self,
                 input_dim=128,
                 num_classes=4,
                 hidden_dims_search_space=[(32, 32), (32, 64), (64, 32), (64, 64)], # Using small archs
                 learning_rate=0.1,
                 batch_size=512,
                 epochs=300,
                 momentum=0.9,
                 # --- MODIFICATIONS ---
                 weight_decay=1e-4, # ADDED: Standard L2 regularization
                 dropout_rate=0.5,  # ADDED: Standard Dropout
                 # --- END MODIFICATIONS ---
                 device="cuda"):

        self.name = "dnn"
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims_search_space = hidden_dims_search_space
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay # <-- MODIFIED
        self.dropout_rate = dropout_rate # <-- ADDED

        use_cuda = (device == "cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.net = None
        self.softmax = nn.Softmax(dim=1)

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _build_net(self, input_dim, hidden_dims):
        """Build a new network with specific input_dim and hidden_dims."""
        self.input_dim = input_dim
        return DNN1D(input_dim=input_dim,
                     num_classes=self.num_classes,
                     hidden_dims=hidden_dims,
                     dropout_rate=self.dropout_rate).to(self.device) # <-- Pass dropout

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def _train_one_model(self, net, X_train, y_train):
        """Internal training loop for one model configuration."""
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True, drop_last=False)

        # --- Using standard (unweighted) loss (as per Run 5) ---
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(net.parameters(),
                                      lr=self.learning_rate,
                                      momentum=self.momentum,
                                      weight_decay=self.weight_decay) # <-- Use weight_decay

        # --- Using LR Scheduler (as per Run 5) ---
        scheduler = MultiStepLR(optimizer,
                                milestones=[self.epochs // 3, (self.epochs * 2) // 3],
                                gamma=0.1)

        net.train()
        for epoch in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

            scheduler.step()

        return net

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    def _evaluate_on_val(self, net, X_val, y_val):
        """Evaluate a trained model on the validation set using Macro F1 (excluding BG)."""
        net.eval() # <-- IMPORTANT: Set model to eval mode for dropout
        with torch.no_grad():
            dl_val = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(self.device)),
                                batch_size=self.batch_size * 2,
                                shuffle=False)
            all_preds = []
            for xb in dl_val:
                logits = net(xb[0])
                all_preds.append(torch.argmax(logits, dim=1).cpu())

            y_pred = torch.cat(all_preds).numpy()

        mask = y_val != 3 # BG class is 3
        if np.sum(mask) == 0:
            return 0.0

        return f1_score(y_val[mask], y_pred[mask], average='macro', labels=[0, 1, 2], zero_division=0)

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    def _predict_proba(self, X):
        """Return softmax probabilities; channels [NT, TT, BV, BG]."""
        if self.net is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        self.net.eval() # <-- IMPORTANT: Set model to eval mode for dropout
        with torch.no_grad():
            dl_pred = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32).to(self.device)),
                                 batch_size=self.batch_size * 2,
                                 shuffle=False)
            all_probs = []
            for xb in dl_pred:
                logits = self.net(xb[0])
                all_probs.append(self.softmax(logits).cpu())

            return torch.cat(all_probs).numpy()

    def predict_full(self, cube):
        """
        Predict class map and probability map for a full HSI cube.
        """
        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T

        if flat.shape[1] != self.input_dim and self.net is not None:
             print(f"[DNNRunner] WARNING: Cube bands ({flat.shape[1]}) != model input ({self.input_dim}).")

        proba = self._predict_proba(flat)
        class_map = np.argmax(proba, axis=1).reshape(H, W)
        prob_all = proba.reshape(H, W, -1)
        return class_map, prob_all
