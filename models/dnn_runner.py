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
    DNN runner con coarse search E early stopping per replicare
    la "stabilizzazione dell'accuratezza" menzionata nel paper.
    """

    def __init__(self,
                 input_dim=128,
                 num_classes=4,
                 hidden_dims_search_space=[(32, 32), (32, 64), (64, 32), (64, 64)],
                 learning_rate=0.1,
                 batch_size=512,
                 epochs=300, # Max epoche
                 momentum=0.9,
                 weight_decay=1e-4,
                 # --- MODIFICATION: Added Early Stopping ---
                 early_stopping_patience=30, # Paper 2019 menziona 40-45 epoche
                 # --- END MODIFICATION ---
                 device="cuda"):

        self.name = "dnn"
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims_search_space = hidden_dims_search_space
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.patience = early_stopping_patience # <-- ADDED

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
    # Evaluation (usato durante l'addestramento)
    # ------------------------------------------------------------
    def _evaluate_on_val(self, net, X_val_tensor, y_val_numpy):
        """Helper per calcolare il Val Macro F1 durante l'addestramento."""
        net.eval()
        with torch.no_grad():
            dl_val = DataLoader(TensorDataset(X_val_tensor),
                                batch_size=self.batch_size * 2,
                                shuffle=False)
            all_preds = []
            for xb in dl_val:
                logits = net(xb[0])
                all_preds.append(torch.argmax(logits, dim=1).cpu())
            y_pred = torch.cat(all_preds).numpy()

        mask = y_val_numpy != 3 # Escludi BG
        if np.sum(mask) == 0:
            return 0.0
        return f1_score(y_val_numpy[mask], y_pred[mask], average='macro', labels=[0, 1, 2], zero_division=0)

    # ------------------------------------------------------------
    # Training (MODIFICATO CON EARLY STOPPING)
    # ------------------------------------------------------------
    def _train_one_model(self, net, X_train, y_train, X_val, y_val):
        """Training loop with validation-based early stopping."""

        # Prepara i Dataloader
        dl_train = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long)),
                              batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Tensore di validazione (per valutazione veloce)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        # Loss pesata (dalla Run 5/8, che dava Val F1 alti)
        cls, cnt = np.unique(y_train, return_counts=True)
        weights = np.zeros(self.num_classes, dtype=np.float32)
        weights[cls] = 1.0 / (cnt + 1e-8)
        weights = torch.tensor(weights / (weights.sum() + 1e-8) * len(weights), dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.SGD(net.parameters(),
                                      lr=self.learning_rate,
                                      momentum=self.momentum,
                                      weight_decay=self.weight_decay)

        scheduler = MultiStepLR(optimizer, milestones=[self.epochs // 3, (self.epochs * 2) // 3], gamma=0.1)

        # Variabili per Early Stopping
        best_val_score = -1.0
        best_model_state = None
        epochs_no_improve = 0
        best_epoch = 0

        for epoch in range(self.epochs):
            net.train() # Attiva BN/Dropout se presenti
            for xb, yb in dl_train:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # --- Valutazione per Early Stopping ---
            val_score = self._evaluate_on_val(net, X_val_tensor, y_val)

            if val_score > best_val_score:
                best_val_score = val_score
                best_model_state = copy.deepcopy(net.state_dict())
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 50 == 0: # Loggiamo i progressi
                 print(f"  [DNNRunner] Epoch {epoch+1}/{self.epochs} | Val F1: {val_score:.4f} (Best: {best_val_score:.4f} at epoch {best_epoch})")

            if epochs_no_improve >= self.patience:
                print(f"  [DNNRunner] Early stopping triggered at epoch {epoch+1}. Best score: {best_val_score:.4f} at epoch {best_epoch}.")
                break

        # Carica il modello migliore
        if best_model_state:
            net.load_state_dict(best_model_state)
        return net, best_val_score # Ritorna anche lo score

    # ------------------------------------------------------------
    # Funzione Fit (MODIFICATA)
    # ------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Esegue la coarse search (ricerca L) e l'addestramento (con early stopping).
        """
        if X_train.size == 0:
            print("[DNNRunner] ⚠ Empty training set, skipping training.")
            return

        if X_val is None or y_val is None or X_val.size == 0:
            # Fallback (non dovrebbe succedere nel nostro script)
            print("[DNNRunner] ⚠ No validation set. Training with default hidden_dims.")
            best_hidden_dims = self.hidden_dims_search_space[0]
            self.net = self._build_net(X_train.shape[1], best_hidden_dims)
            self.net, _ = self._train_one_model(self.net, X_train, y_train, X_train, y_train) # Usa train per val
            return

        print(f"[DNNRunner] Starting coarse search for L over {self.hidden_dims_search_space}...")
        best_overall_score = -1.0
        best_net = None
        best_hidden_dims = None

        for hidden_dims in self.hidden_dims_search_space:
            print(f"[DNNRunner] Testing L = {hidden_dims}...")
            current_net = self._build_net(X_train.shape[1], hidden_dims)

            # _train_one_model ora fa l'addestramento E l'early stopping
            current_net, score = self._train_one_model(current_net, X_train, y_train, X_val, y_val)

            print(f"[DNNRunner]   L = {hidden_dims} | Final Val Macro F1 = {score:.4f}")

            if score > best_overall_score:
                best_overall_score = score
                best_net = copy.deepcopy(current_net) # Salva il modello già addestrato
                best_hidden_dims = hidden_dims

        self.net = best_net
        self.input_dim = X_train.shape[1]

        print(f"[DNNRunner] ✅ Best L found: {best_hidden_dims} (Score: {best_overall_score:.4f})")

    # ------------------------------------------------------------
    # Prediction (MODIFICATA per usare _evaluate_on_val)
    # ------------------------------------------------------------
    def _predict_proba(self, X):
        """Return softmax probabilities; channels [NT, TT, BV, BG]."""
        if self.net is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        self.net.eval()
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
