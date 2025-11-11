# models/nebae_runner.py
"""
NEBEAERunner (Paper-Compliant)
-----------
Implements the NEBEAE (Nonlinear Extended Blind End-member and Abundance
Extraction) unmixing approach used in the HSI benchmark[cite: 935].

This implementation is based on the Cyclic Coordinate Descent (CCD)
framework for a Multilinear Mixture Model (MMM) (e.g., bilinear)
described in the cited paper [48].
"""

import numpy as np
from . import BaseRunner
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from scipy.optimize import minimize

# Disabilitiamo i warning di runtime
np.seterr(divide='ignore', invalid='ignore')


class NEBEAERunner(BaseRunner):
    """
    NEBEAE unmixing-based model runner.

    Implements the CCD framework for a bilinear model
    with paper-specific hyperparameters[cite: 938, 939, 940].
    """

    def __init__(self,
                 y_search_space=[0.001, 0.01, 0.1, 1.0],
                 max_ccd_iter=30,
                 ccd_tol=1e-5,
                 random_state=42):

        self.name = "nebeae"

        # Parametri fissi dal paper (identici a EBEAE)
        self.class_endmembers_count = {0: 2, 1: 2, 2: 1, 3: 3} # NT, TT, BV, BG
        self.class_p = {0: 0.3, 1: 0.2, 2: 0.0, 3: 0.01} #

        # Parametri di ottimizzazione
        self.y_search_space = y_search_space #
        self.max_ccd_iter = max_ccd_iter
        self.ccd_tol = ccd_tol
        self.random_state = random_state

        # Risultati del fit
        self.endmembers_ = None           # Matrice Lineare (K_total, B)
        self.nonlinear_terms_ = None    # Matrice Non-Lineare (K_total, B)
        self.endmember_class_map_ = None  # Array (K_total,)
        self.best_y_ = 0                  # y ottimale trovato

        # Etichette per la metrica F1 (esclude BG, label 3) [cite: 976]
        self.metric_labels = [0, 1, 2]


    # ------------------------------------------------------------
    # FIT (Ottimizzazione Iperparametro 'y')
    # ------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Esegue il fit. Trova i parametri E e B usando CCD (con p),
        poi ottimizza 'y' (entropia) sul set di validazione.
        """
        if X_train.size == 0:
            print("[NEBEAE] ⚠ Empty training set, skipping training.")
            return

        print(f"[NEBEAE] 1. Fitting Model (E, B) using CCD...")
        E_fit, B_fit, class_map = self._find_model_params(X_train, y_train)

        print(f"[NEBEAE] 2. Optimizing 'y' (entropy weight) on validation set...")

        best_score = -1.0
        best_y = 0.0

        if X_val is None or X_val.size == 0:
             print("[NEBEAE] ⚠ No validation set, skipping 'y' optimization.")
             self.best_y_ = self.y_search_space[0]
             self.endmembers_ = E_fit
             self.nonlinear_terms_ = B_fit
             self.endmember_class_map_ = class_map
             return

        for y_current in self.y_search_space:
            # Stima le abbondanze sul set di validazione usando 'y'
            A_val = self._find_abundances_nonlinear(X_val, E_fit, B_fit, y_current)

            # Converti le abbondanze in etichette
            y_pred_val = self._abundances_to_labels(A_val, class_map)

            # Calcola F1-Score (escludendo BG) [cite: 949, 976]
            score = f1_score(
                y_val,
                y_pred_val,
                labels=self.metric_labels,
                average="macro",
                zero_division=0.0
            )
            print(f"[NEBEAE]   y = {y_current:.4f}, Val F1 = {score:.4f}")

            if score > best_score:
                best_score = score
                self.best_y_ = y_current
                self.endmembers_ = E_fit
                self.nonlinear_terms_ = B_fit
                self.endmember_class_map_ = class_map

        print(f"[NEBEAE] ✅ Fit complete. Best y = {self.best_y_} (Val F1={best_score:.4f})")


    # ------------------------------------------------------------
    # PREDICTION (Usa 'y' ottimale)
    # ------------------------------------------------------------

    def predict_full(self, cube):
        """
        Esegue l'unmixing (stima delle abbondanze) sull'intero cubo
        utilizzando i parametri E, B trovati e il 'y' ottimale.
        """
        if self.endmembers_ is None:
            raise RuntimeError("[NEBEAE] ❌ Model not trained. Call .fit() first.")

        bands, H, W = cube.shape
        M_flat = cube.reshape(bands, -1).T # (N_pixels, N_bands)

        print(f"[NEBEAE] Predicting on cube ({bands} bands, {H}×{W} spatial) using y={self.best_y_}...")

        # A è (N_pixels, K_total)
        A_flat = self._find_abundances_nonlinear(
            M_flat,
            self.endmembers_,
            self.nonlinear_terms_,
            self.best_y_
        )

        # Converte abbondanze in etichette
        class_map, prob_all = self._abundances_to_class_probs(A_flat, self.endmember_class_map_, self.class_endmembers_count.keys())

        print("[NEBEAE] ✅ Prediction complete.")
        return class_map.reshape(H, W), prob_all.reshape(H, W, -1)


    # ------------------------------------------------------------
    # CORE: Stima Modello (Fase 1 del Fit)
    # ------------------------------------------------------------

    def _find_model_params(self, X, y):
        """
        Trova i parametri E e B usando l'approccio CCD.
        X e y sono X_train e y_train.
        """

        all_E = []
        all_class_map = []
        B_bands = X.shape[1] # Numero di bande

        # 1. Inizializzazione Endmember (E) e Termini Non Lineari (B)
        for c, count in self.class_endmembers_count.items():
            X_c = X[y == c]

            if X_c.shape[0] == 0:
                print(f"[NEBEAE] WARNING: No training pixels for class {c}. Initializing endmember(s) to zero.")
                E_c = np.zeros((count, B_bands))
            elif c == 2: # Caso speciale BV
                E_c = np.mean(X_c, axis=0, keepdims=True)
            else:
                # Inizializza E con K-Means
                n_init = min(count, X_c.shape[0])
                km = KMeans(n_clusters=n_init, n_init=3, random_state=self.random_state)
                E_c = km.fit(X_c).cluster_centers_
                if E_c.shape[0] < count:
                    E_c = np.vstack([E_c, np.zeros((count - E_c.shape[0], B_bands))])

            all_E.append(E_c)
            all_class_map.append(np.full(E_c.shape[0], c))

        E = np.vstack(all_E)
        class_map = np.concatenate(all_class_map)
        B_nl = np.zeros_like(E) # Inizializza i termini non lineari a zero

        # Salva E e B iniziali in caso di fallimento
        self.endmembers_ = E
        self.nonlinear_terms_ = B_nl
        self.endmember_class_map_ = class_map

        # 2. Loop CCD (Cyclic Coordinate Descent)
        for i in range(self.max_ccd_iter):
            # A. Stima Abbondanze (A)
            A = self._find_abundances_nonlinear(X, E, B_nl, y_entropy_weight=0.0)
            A_sq = A**2

            # B. Stima Endmember Lineari (E)
            # Risolvi A @ E = X - (A**2) @ B_nl
            X_resid_E = X - (A_sq @ B_nl)
            E_new = self._update_parameters_als(X_resid_E, A, class_map, self.class_p, X, y)

            # C. Stima Termini Non Lineari (B)
            # Risolvi (A**2) @ B_nl = X - A @ E
            X_resid_B = X - (A @ E_new)
            B_new = self._update_parameters_als(X_resid_B, A_sq, class_map, self.class_p, X, y, is_bv_special_case=False)

            # Controllo Convergenza
            diff_E = np.linalg.norm(E - E_new, 'fro') / (np.linalg.norm(E, 'fro') + 1e-9)
            diff_B = np.linalg.norm(B_nl - B_new, 'fro') / (np.linalg.norm(B_nl, 'fro') + 1e-9)

            E = E_new
            B_nl = B_new

            if (diff_E < self.ccd_tol) and (diff_B < self.ccd_tol):
                print(f"[NEBEAE] CCD converged at iteration {i+1}.")
                break

        return E, B_nl, class_map


    def _update_parameters_als(self, X_target, A_features, class_map, class_p, X_train, y_train, is_bv_special_case=True):
        """
        Funzione helper generica per ALS/CCD (Passo B e C).
        Risolve P = (A_features.T @ A_features + p*I)^-1 @ A_features.T @ X_target
        """
        K_total = A_features.shape[1]
        B_bands = X_target.shape[1]
        P_new = np.zeros((K_total, B_bands))

        for c in np.unique(class_map):
            idx = np.where(class_map == c)[0]
            if idx.size == 0: continue

            p = class_p.get(c, 0.0)
            K_c = idx.size

            # Il caso speciale BV si applica solo al termine Lineare E (Passo B)
            if c == 2 and is_bv_special_case:
                X_c_train = X_train[y_train == c]
                if X_c_train.shape[0] > 0:
                    P_new[idx] = np.mean(X_c_train, axis=0, keepdims=True)
                else:
                    pix_idx_bv = np.argmax(A_features[:, idx[0]])
                    P_new[idx] = X_train[pix_idx_bv]
            else:
                # Logica ALS/CCD standard
                A_c = A_features[:, idx]
                A_c_T = A_c.T
                left = A_c_T @ A_c + p * np.eye(K_c)
                right = A_c_T @ X_target

                try:
                    P_new[idx] = np.linalg.solve(left, right)
                except np.linalg.LinAlgError:
                    P_new[idx] = np.linalg.pinv(left) @ right

        P_new[P_new < 0] = 0
        return P_new

    # ------------------------------------------------------------
    # CORE: Stima Abbondanze Non Lineare
    # ------------------------------------------------------------

    def _find_abundances_nonlinear(self, X_pixels, E, B, y_entropy_weight):
        """
        Trova le abbondanze A per tutti i pixel, dati E, B, e y.
        (Questa è la parte computazionalmente MOLTO LENTA)
        """
        N_pixels = X_pixels.shape[0]
        K_total = E.shape[0]
        A_out = np.zeros((N_pixels, K_total))

        constraints = ({'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0})
        bounds = tuple((0, None) for _ in range(K_total))
        a_init = np.full(K_total, 1.0 / K_total)

        for i in range(N_pixels):
            m = X_pixels[i] # Pixel corrente (1, B_bands)

            def objective(a):
                # a è (K_total,)
                a_sq = a**2

                # Modello Bilineare: X_pred = a @ E + a_sq @ B
                # E e B sono (K_total, B_bands)
                x_pred = a @ E + a_sq @ B # (1, B_bands)

                # Termine 1: Errore di ricostruzione
                recon_error = 0.5 * np.sum((x_pred - m)**2)

                # Termine 2: Entropia (regolarizzazione 'y')
                a_log = a.copy()
                a_log[a_log < 1e-9] = 1e-9
                entropy = -np.sum(a * np.log(a_log))

                return recon_error - y_entropy_weight * entropy

            res = minimize(objective, a_init, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 100, 'ftol': 1e-6})

            if res.success:
                A_out[i] = res.x
            else:
                A_out[i] = a_init

        A_out[A_out < 0] = 0
        return A_out


    # ------------------------------------------------------------
    # Helpers (Identici a EBEAE)
    # ------------------------------------------------------------

    def _abundances_to_class_probs(self, A_flat, class_map, class_labels):
        N_pixels, K_total = A_flat.shape
        N_classes = len(class_labels)
        prob_all = np.zeros((N_pixels, N_classes))

        for c in class_labels:
            class_indices = np.where(class_map == c)[0]
            if class_indices.size > 0:
                prob_all[:, c] = np.sum(A_flat[:, class_indices], axis=1)

        row_sums = prob_all.sum(axis=1, keepdims=True)
        prob_all = prob_all / (row_sums + 1e-8)
        class_map = np.argmax(prob_all, axis=1)

        return class_map, prob_all

    def _abundances_to_labels(self, A_flat, class_map):
        class_map_pred, _ = self._abundances_to_class_probs(
            A_flat,
            class_map,
            self.class_endmembers_count.keys()
        )
        return class_map_pred
