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
import copy 
from sklearn.metrics import f1_score

class DNN1D(nn.Module):
    def __init__(self, input_dim=128, num_classes=4, hidden_dims=(256, 256)):
        super().__init__()
        assert len(hidden_dims) == 2, "This paper-aligned DNN expects exactly two hidden layers."

        h1, h2 = hidden_dims

        # Layer 1: Linear -> ReLU
        layer1 = [
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
        ]

        # Layer 2: Linear -> ReLU
        layer2 = [
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
        ]
        
        # Batch Norm *follows* the two hidden layers, as per paper description
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
                 # Define the "coarse search" space for L (hidden_dims) as per paper
                 hidden_dims_search_space=[(128, 128), (256, 256), (512, 512)], # <-- MODIFIED
                 learning_rate=0.1,
                 batch_size=512,
                 epochs=300,
                 momentum=0.9,
                 weight_decay=0.0,
                 device="cuda"):
        
        self.name = "dnn"
        self.input_dim = input_dim # Will be updated by fit()
        self.num_classes = num_classes
        self.hidden_dims_search_space = hidden_dims_search_space
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay

        use_cuda = (device == "cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.net = None # Will be set by fit()
        self.softmax = nn.Softmax(dim=1)

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    def _build_net(self, input_dim, hidden_dims): # <-- RENAMED from _maybe_rebuild
        """Build a new network with specific input_dim and hidden_dims."""
        self.input_dim = input_dim
        return DNN1D(input_dim=input_dim,
                     num_classes=self.num_classes,
                     hidden_dims=hidden_dims).to(self.device)

    def _train_one_model(self, net, X_train, y_train): # <-- NEW HELPER
        """Internal training loop for one model configuration."""
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True, drop_last=False)
        
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=self.learning_rate,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        
        net.train()
        # Suppress epoch logging during coarse search
        # print_every = self.epochs // 3
        
        for epoch in range(self.epochs):
            # total_loss = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()
                # total_loss += loss.item()
            
            # if (epoch + 1) % print_every == 0 or epoch == 0:
            #     print(f"  [DNNRunner] Epoch {epoch+1}/{self.epochs} | Loss: {total_loss/len(dl):.4f}")
                
        return net # Return the trained net

    def _evaluate_on_val(self, net, X_val, y_val): # <-- NEW HELPER
        """Evaluate a trained model on the validation set using Macro F1 (excluding BG)."""
        net.eval()
        with torch.no_grad():
            # Predict in batches to avoid OOM
            dl_val = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32).to(self.device)),
                                batch_size=self.batch_size * 2, 
                                shuffle=False)
            all_preds = []
            for xb in dl_val:
                logits = net(xb[0])
                all_preds.append(torch.argmax(logits, dim=1).cpu())
            
            y_pred = torch.cat(all_preds).numpy()
        
        # Use Macro F1-Score as per paper (excluding BG class 3)
        mask = y_val != 3 # BG class is 3
        if np.sum(mask) == 0:
            return 0.0 # No non-BG samples to evaluate
            
        # Evaluate F1 only on NT(0), TT(1), BV(2)
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
            # No validation set, just train with the first L in search space
            print("[DNNRunner] ⚠ No validation set. Training with default hidden_dims.")
            best_hidden_dims = self.hidden_dims_search_space[0]
            self.net = self._build_net(X_train.shape[1], best_hidden_dims)
            self._train_one_model(self.net, X_train, y_train)
            return

        # --- Coarse Search Loop ---
        print(f"[DNNRunner] Starting coarse search for L over {self.hidden_dims_search_space}...")
        best_score = -1.0
        best_net = None
        best_hidden_dims = None

        for hidden_dims in self.hidden_dims_search_space:
            # 1. Create model
            current_net = self._build_net(X_train.shape[1], hidden_dims)
            # 2. Train model
            current_net = self._train_one_model(current_net, X_train, y_train)
            # 3. Evaluate on validation set
            score = self._evaluate_on_val(current_net, X_val, y_val)
            print(f"[DNNRunner]   L = {hidden_dims} | Val Macro F1 = {score:.4f}")

            if score > best_score:
                best_score = score
                best_net = copy.deepcopy(current_net) # Store the best model state
                best_hidden_dims = hidden_dims
        
        # Set the runner's final model to the best one
        self.net = best_net
        self.input_dim = X_train.shape[1] # Ensure input_dim is set
        
        print(f"[DNNRunner] ✅ Best L found: {best_hidden_dims} (Score: {best_score:.4f})")

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    def _predict_proba(self, X):
        """Return softmax probabilities; channels [NT, TT, BV, BG]."""
        if self.net is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        
        self.net.eval()
        with torch.no_grad():
            # Predict in batches
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
        Returns:
            class_map: (H, W) in {0:NT,1:TT,2:BV,3:BG}
            prob_all:  (H, W, num_classes) with channel order [NT, TT, BV, BG]
        """
        bands, H, W = cube.shape
        flat = cube.reshape(bands, -1).T
        
        if flat.shape[1] != self.input_dim and self.net is not None:
             print(f"[DNNRunner] WARNING: Cube bands ({flat.shape[1]}) != model input ({self.input_dim}).")

        proba = self._predict_proba(flat)
        class_map = np.argmax(proba, axis=1).reshape(H, W)
        prob_all = proba.reshape(H, W, -1)
        return class_map, prob_all
