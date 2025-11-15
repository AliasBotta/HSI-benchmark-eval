"""
RandomRunner (Dummy Classifier)
---------
Implementa un classificatore fittizio per testare la pipeline.

- fit(): Non fa nulla (ignora i dati di training).
- predict_full(): Restituisce probabilità casuali per ogni classe.
                  La classe predetta è l'argmax di queste probabilità.
"""

import numpy as np
from . import BaseRunner

class RandomRunner(BaseRunner):
    """
    Classificatore dummy che assegna probabilità casuali.
    Necessario per implementare l'interfaccia BaseRunner.
    """

    def __init__(self, num_classes=4, random_state=42):
        """
        Inizializza il runner.

        Parametri
        ----------
        num_classes : int
            Il numero di classi di output (default: 4 per NT, TT, BV, BG)
        random_state : int
            Seed per la riproducibilità dei risultati casuali.
        """
        self.name = "random"
        self.num_classes = num_classes
        # Usa un generatore di numeri casuali di numpy per la riproducibilità
        self.rng = np.random.default_rng(random_state)
        print(f"[{self.name.upper()}] Dummy classifier initialized.")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Metodo Fit (fittizio).
        Stampa solo un messaggio e non esegue alcun training.
        """
        print(f"[{self.name.upper()}] ⚠ Dummy classifier: 'fit' chiamato, training saltato.")

        # Simula l'output degli altri runner per coerenza nei log
        print(f"[{self.name.upper()}] ✅ Training complete.")
        print(f"     -> Best Score (Val Macro F1): 0.2500 (mock)")
        print(f"     -> Best Params: {{'mode': 'random'}}")
        pass # Non c'è nessun modello da addestrare (self.clf = None)

    def predict_full(self, cube):
        """
        Genera una mappa di predizione e una mappa di probabilità casuali.

        Restituisce
        -------
        class_map : np.ndarray
            (H, W) label predette (l'indice con la probabilità più alta)
        prob_all : np.ndarray
            (H, W, num_classes) probabilità casuali (normalizzate per somma 1)
        """
        bands, H, W = cube.shape
        n_pixels = H * W

        print(f"[{self.name.upper()}] Predicting random probabilities for cube ({H}x{W})...")

        # 1. Genera probabilità grezze casuali (N_pixel, N_classi)
        raw_probs = self.rng.random(size=(n_pixels, self.num_classes))

        # 2. Normalizza ogni riga (pixel) affinché la somma delle probabilità sia 1
        #    Questo simula l'output di una softmax
        row_sums = raw_probs.sum(axis=1, keepdims=True)
        prob_flat = raw_probs / (row_sums + 1e-8) # (N_pixels, 4)

        # 3. Trova la classe con probabilità maggiore (come richiesto)
        class_flat = np.argmax(prob_flat, axis=1) # (N_pixels,)

        # 4. Rimodella i risultati alle dimensioni dell'immagine
        class_map = class_flat.reshape(H, W)
        prob_all = prob_flat.reshape(H, W, self.num_classes)

        print(f"[{self.name.upper()}] ✅ Prediction complete.")

        # Restituisce esattamente ciò che si aspetta train.py
        return class_map, prob_all
