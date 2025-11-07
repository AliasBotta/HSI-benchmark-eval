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
    # --- RIPRISTINATA ARCHITETTURA 'L2 -> BN -> ReLU' (dalla Run 5) ---
    def __init__(self, input_dim=128, num_classes=4, hidden_dims=(256, 256)):
        super().__init__()
        assert len(hidden_dims) == 2, "This paper-aligned DNN expects exactly two hidden layers."

        h1, h2 = hidden_dims

        layer1 = [
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
        ]
        layer2 = [
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2), # BN prima della ReLU
            nn.ReLU(inplace=True),
        ]
        out = [nn.Linear(h2, num_classes)]
        self.net = nn.Sequential(*(layer1 + layer2 + out))

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
                 hidden_dims_search_space=[(32, 32), (32, 64), (64, 32), (64, 64)], # Spazio di ricerca piccolo
                 learning_rate=0.1,
                 batch_size=512,
                 epochs=300,
                 momentum=0.9,
                 weight_decay=1e-4, # Aggiunta L2 regularization
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
        # Rimosso Dropout, il BN è sufficiente

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
                     hidden_dims=hidden_dims).to(self.device)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def _train_one_model(self, net, X_train, y_train):
        """Internal training loop for one model configuration."""
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True, drop_last=False)

        # --- RIPRISTINATI I PESI DELLA LOSS (dalla Run 5) ---
        cls, cnt = np.unique(y_train, return_counts=True)
        weights = np.zeros(self.num_classes, dtype=np.float32)
        weights[cls] = 1.0 / (cnt + 1e-8)
        weights = torch.tensor(weights / (weights.sum() + 1e-8) * len(weights), dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        # --- FINE RIPRISTINO PESI ---

        optimizer = torch.optim.SGD(net.parameters(),
                                      lr=self.learning_rate,
                                      momentum=self.momentum,
                                      weight_decay=self.weight_decay) # <-- Usa weight_decay

        # --- Aggiunto LR Scheduler ---
        scheduler = MultiStepLR(optimizer, milestones=[self.epochs // 3, (self.epochs * 2) // 3], gamma=0.1)

        net.train() # Set model to training mode

        for epoch in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

            scheduler.step() # Step per epoch

        return net

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    def _evaluate_on_val(self, net, X_val, y_val):
        """Evaluate a trained model on the validation set using Macro F1 (excluding BG)."""
        net.eval() # <-- Set model to eval mode
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
    # Training (Now with Coarse Search)
    # ------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the DNN. If X_val, y_val are provided, performs coarse search
        for the best hidden_dims (L) from hidden_dims_search_space.
        """
        if X_train.size == 0:
            print("[DNNRunner] ⚠ Empty training set, skipping training.")
            return

        if X_val is None or y_val is None or X_val.size == 0:
            print("[DNNRunner] ⚠ No validation set. Training with default hidden_dims.")
            best_hidden_dims = self.hidden_dims_search_space[0]
            self.net = self._build_net(X_train.shape[1], best_hidden_dims)
            self._train_one_model(self.net, X_train, y_train)
            return

        print(f"[DNNRunner] Starting coarse search for L over {self.hidden_dims_search_space}...")
        best_score = -1.0
        best_net = None
        best_hidden_dims = None

        for hidden_dims in self.hidden_dims_search_space:
            current_net = self._build_net(X_train.shape[1], hidden_dims)
            current_net = self._train_one_model(current_net, X_train, y_train)
            score = self._evaluate_on_val(current_net, X_val, y_val)
            print(f"[DNNRunner]   L = {hidden_dims} | Val Macro F1 = {score:.4f}")

            if score > best_score:
                best_score = score
                best_net = copy.deepcopy(current_net)
                best_hidden_dims = hidden_dims

        self.net = best_net
        self.input_dim = X_train.shape[1]

        print(f"[DNNRunner] ✅ Best L found: {best_hidden_dims} (Score: {best_score:.4f})")

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    def _predict_proba(self, X):
        """Return softmax probabilities; channels [NT, TT, BV, BG]."""
        if self.net is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        self.net.eval() # <-- Set model to eval mode
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
