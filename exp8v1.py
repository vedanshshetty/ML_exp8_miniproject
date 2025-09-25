import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ---------- plotting helpers ----------
def plot_confusion_matrix(cm, class_labels, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {title}")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, model_name, label_type):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} ({label_type})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# ---------- stacking (SFE) helpers ----------
def fit_stacking_clusterers(X_train, n_clusters_kmeans=8, n_components_gmm=8,
                            random_state=42, use_minibatch=True, gmm_sample_max=200000):
    """
    Fit KMeans and GMM on X_train (only). Return fitted kmeans and gmm objects.
    For big X_train, GMM is fit on a random sample for speed.
    """
    if use_minibatch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters_kmeans, random_state=random_state,
                                 batch_size=4096, n_init=3)
    else:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=random_state, n_init=10)
    kmeans.fit(X_train)

    # fit GMM on a sample if X_train large
    n_rows = X_train.shape[0]
    if n_rows > gmm_sample_max:
        idx = np.random.choice(n_rows, size=gmm_sample_max, replace=False)
        gmm_fit_data = X_train[idx]
    else:
        gmm_fit_data = X_train

    gmm = GaussianMixture(n_components=n_components_gmm, random_state=random_state, covariance_type='diag')
    gmm.fit(gmm_fit_data)

    return kmeans, gmm

def transform_stacking_features(X, kmeans, gmm):
    """
    Given fitted kmeans and gmm, compute meta-features for X.
    Returns a 2D numpy array.
    """
    k_labels = kmeans.predict(X).reshape(-1, 1)
    k_dist = kmeans.transform(X)                         # distance to kmeans centers -> (n_samples, n_clusters)
    gmm_probs = gmm.predict_proba(X)                     # (n_samples, n_components)
    gmm_labels = gmm.predict(X).reshape(-1, 1)
    meta = np.hstack([k_labels, gmm_labels, k_dist, gmm_probs])
    return meta

# ---------- pipeline ----------
class IntrusionDetectionPipeline:
    def __init__(self, random_state=42, n_splits=5, n_pca=20,
                 n_clusters_kmeans=8, n_components_gmm=8,
                 use_minibatch=True):
        self.random_state = random_state
        self.n_splits = n_splits
        self.n_pca = n_pca
        self.n_clusters_kmeans = n_clusters_kmeans
        self.n_components_gmm = n_components_gmm
        self.use_minibatch = use_minibatch

        # models to evaluate
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=-1)
        }

        # placeholders for encoders and scalers
        self.cat_encoders = {}
        self.attack_cat_encoder = None
        self.scaler = None

    def load_data(self, train_csv, test_csv):
        print(f"Loading: {train_csv} and {test_csv}")
        self.df_train_raw = pd.read_csv(train_csv)
        self.df_test_raw = pd.read_csv(test_csv)

    # ------------ Preprocessing ------------
    def preprocess_train(self, df):
        """
        Fit encoders on train and return processed df and fitted encoders.
        """
        df = df.copy()
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # numeric downcast
        for c in df.select_dtypes(include=['int64']).columns:
            df[c] = pd.to_numeric(df[c], downcast='integer')
        for c in df.select_dtypes(include=['float64']).columns:
            df[c] = pd.to_numeric(df[c], downcast='float')

        # categorical encoding: fit and store encoders
        self.cat_encoders = {}
        for c in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            self.cat_encoders[c] = le

        if 'attack_cat' in df.columns:
            self.attack_cat_encoder = LabelEncoder()
            df['attack_cat_enc'] = self.attack_cat_encoder.fit_transform(df['attack_cat'].astype(str))

        return df

    def preprocess_apply(self, df):
        """
        Apply fitted encoders to new dataframe (test or val). Unseen categories mapped to -1.
        """
        df = df.copy()
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        for c in df.select_dtypes(include=['int64']).columns:
            df[c] = pd.to_numeric(df[c], downcast='integer')
        for c in df.select_dtypes(include=['float64']).columns:
            df[c] = pd.to_numeric(df[c], downcast='float')

        for c, le in self.cat_encoders.items():
            if c in df.columns:
                vals = df[c].astype(str).values
                # map unseen -> -1
                mapped = np.array([np.where(le.classes_ == v)[0][0] if v in le.classes_ else -1 for v in vals])
                df[c] = mapped.astype(int)

        if 'attack_cat' in df.columns and self.attack_cat_encoder is not None:
            vals = df['attack_cat'].astype(str).values
            mapped = np.array([np.where(self.attack_cat_encoder.classes_ == v)[0][0] if v in self.attack_cat_encoder.classes_ else -1 for v in vals])
            df['attack_cat_enc'] = mapped.astype(int)

        return df

    # ------------ Run pipeline ------------
    def run(self, df_train_raw=None, df_test_raw=None):
        if df_train_raw is None:
            df_train_raw = self.df_train_raw
        if df_test_raw is None:
            df_test_raw = self.df_test_raw

        print("Preprocessing train & test with shared encoders...")
        df_train = self.preprocess_train(df_train_raw)
        df_test = self.preprocess_apply(df_test_raw)

        # determine targets and features
        target_binary = 'label'
        target_multi = 'attack_cat_enc' if 'attack_cat_enc' in df_train.columns else None

        exclude_cols = [target_binary]
        if target_multi:
            exclude_cols += ['attack_cat', 'attack_cat_enc']
        else:
            if 'attack_cat' in df_train.columns:
                exclude_cols += ['attack_cat']

        features = [c for c in df_train.columns if c not in exclude_cols]

        # scale (fit on training only)
        self.scaler = StandardScaler()
        X_train_all = self.scaler.fit_transform(df_train[features])
        X_test_all = self.scaler.transform(df_test[features])

        y_train_bin = df_train[target_binary].values
        y_test_bin = df_test[target_binary].values

        y_train_multi = df_train[target_multi].values if target_multi else None
        y_test_multi = df_test[target_multi].values if target_multi else None

        # First: run CV (evaluate on validation folds)
        print("Starting cross-validation to validate models (metrics from validation sets)...")
        cv_results = self.cross_validate(X_train_all, y_train_bin, label_type="Binary")

        if y_train_multi is not None:
            cv_results_multi = self.cross_validate(X_train_all, y_train_multi, label_type="Multiclass")
        else:
            cv_results_multi = None

        # After CV: final training on full training data and evaluate once on held-out test set
        print("\nTraining on full training set and evaluating on external test set (final evaluation)...")
        final_results = self.train_and_evaluate_on_test(X_train_all, y_train_bin, X_test_all, y_test_bin, label_type="Binary")

        if y_train_multi is not None:
            final_results_multi = self.train_and_evaluate_on_test(X_train_all, y_train_multi, X_test_all, y_test_multi, label_type="Multiclass")
        else:
            final_results_multi = None

        return {
            'cv_binary': cv_results,
            'final_binary': final_results,
            'cv_multi': cv_results_multi,
            'final_multi': final_results_multi
        }

    # ------------ Cross-validation (validation metrics) ------------
    def cross_validate(self, X, y, label_type="Binary"):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        results = {name: {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'roc': [], 'cm': None} for name in self.models.keys()}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n=== Fold {fold}/{self.n_splits} ===")
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Oversample training fold only
            ros = RandomOverSampler(random_state=self.random_state)
            X_tr_os, y_tr_os = ros.fit_resample(X_tr, y_tr)
            print(f"  Oversampled from {X_tr.shape[0]} -> {X_tr_os.shape[0]} samples")

            # Fit clusterers on training fold (oversampled)
            t0 = time.time()
            kmeans, gmm = fit_stacking_clusterers(X_tr_os, n_clusters_kmeans=self.n_clusters_kmeans,
                                                 n_components_gmm=self.n_components_gmm,
                                                 random_state=self.random_state,
                                                 use_minibatch=self.use_minibatch)
            t1 = time.time()
            print(f"  Fitted clusterers in {t1 - t0:.2f}s")

            # Transform to meta-features (no refit)
            sfe_tr = transform_stacking_features(X_tr_os, kmeans, gmm)
            sfe_val = transform_stacking_features(X_val, kmeans, gmm)

            # Augment features
            X_tr_aug = np.hstack([X_tr_os, sfe_tr])
            X_val_aug = np.hstack([X_val, sfe_val])

            # PCA on training augmented only
            n_pca_comp = min(self.n_pca, X_tr_aug.shape[1])
            pca = PCA(n_components=n_pca_comp, random_state=self.random_state)
            X_tr_pca = pca.fit_transform(X_tr_aug)
            X_val_pca = pca.transform(X_val_aug)

            # For each model, train and evaluate on validation
            for name, model in self.models.items():
                print(f"  Training {name} ...", end="")
                m = model.__class__(**model.get_params())  # fresh instance
                m.fit(X_tr_pca, y_tr_os)
                y_val_pred = m.predict(X_val_pca)

                # try predict_proba for ROC if possible
                y_prob = None
                try:
                    y_prob = m.predict_proba(X_val_pca)
                except Exception:
                    y_prob = None

                average = 'macro' if label_type == "Multiclass" else 'binary'
                acc = accuracy_score(y_val, y_val_pred)
                prec = precision_score(y_val, y_val_pred, average=average, zero_division=0)
                rec = recall_score(y_val, y_val_pred, average=average, zero_division=0)
                f1 = f1_score(y_val, y_val_pred, average=average, zero_division=0)

                # ROC AUC handling
                if y_prob is not None:
                    try:
                        if label_type == "Multiclass":
                            roc_val = roc_auc_score(y_val, y_prob, multi_class='ovr', average='macro')
                        else:
                            # binary
                            if y_prob.shape[1] == 2:
                                roc_val = roc_auc_score(y_val, y_prob[:, 1])
                            else:
                                # fallback if single-proba provided
                                roc_val = np.nan
                    except Exception:
                        roc_val = np.nan
                else:
                    roc_val = np.nan

                results[name]['acc'].append(acc)
                results[name]['prec'].append(prec)
                results[name]['rec'].append(rec)
                results[name]['f1'].append(f1)
                results[name]['roc'].append(roc_val)

                # accumulate confusion matrix across folds
                cm = confusion_matrix(y_val, y_val_pred)
                if results[name]['cm'] is None:
                    results[name]['cm'] = cm
                else:
                    # ensure shapes match (they should)
                    results[name]['cm'] += cm

                print(f" done. Acc={acc:.4f} F1={f1:.4f} ROC={roc_val if not np.isnan(roc_val) else 'nan'}")

        # aggregate results
        summary = {}
        for name, stats in results.items():
            summary[name] = {
                'accuracy_mean': np.mean(stats['acc']),
                'precision_mean': np.mean(stats['prec']),
                'recall_mean': np.mean(stats['rec']),
                'f1_mean': np.mean(stats['f1']),
                'roc_mean': np.nanmean(stats['roc']),
                'confusion_matrix': stats['cm']
            }
            print(f"\n{name} CV summary - Acc: {summary[name]['accuracy_mean']:.4f}, "
                  f"Prec: {summary[name]['precision_mean']:.4f}, "
                  f"Rec: {summary[name]['recall_mean']:.4f}, "
                  f"F1: {summary[name]['f1_mean']:.4f}, "
                  f"ROC: {summary[name]['roc_mean']:.4f}")

        return summary

    # ------------ Final training on full train set & test evaluation ------------
    def train_and_evaluate_on_test(self, X_train_all, y_train_all, X_test_all, y_test, label_type="Binary"):
        """
        Fit clusterers + PCA + model on full training data (optionally oversampled),
        then evaluate only once on the external test set.
        """
        # Oversample entire training (optional; paper used oversampling)
        ros = RandomOverSampler(random_state=self.random_state)
        X_tr_os, y_tr_os = ros.fit_resample(X_train_all, y_train_all)
        print(f"  Oversampled full train: {X_train_all.shape[0]} -> {X_tr_os.shape[0]}")

        # Fit clusterers on oversampled full training
        kmeans, gmm = fit_stacking_clusterers(X_tr_os, n_clusters_kmeans=self.n_clusters_kmeans,
                                             n_components_gmm=self.n_components_gmm,
                                             random_state=self.random_state,
                                             use_minibatch=self.use_minibatch)

        # Transform
        sfe_tr = transform_stacking_features(X_tr_os, kmeans, gmm)
        sfe_test = transform_stacking_features(X_test_all, kmeans, gmm)

        X_tr_aug = np.hstack([X_tr_os, sfe_tr])
        X_test_aug = np.hstack([X_test_all, sfe_test])

        # PCA on full
        n_pca_comp = min(self.n_pca, X_tr_aug.shape[1])
        pca = PCA(n_components=n_pca_comp, random_state=self.random_state)
        X_tr_pca = pca.fit_transform(X_tr_aug)
        X_test_pca = pca.transform(X_test_aug)

        final_summary = {}
        for name, model in self.models.items():
            print(f"\nTraining final {name} on full training set...")
            m = model.__class__(**model.get_params())
            m.fit(X_tr_pca, y_tr_os)
            y_test_pred = m.predict(X_test_pca)

            y_prob = None
            try:
                y_prob = m.predict_proba(X_test_pca)
            except Exception:
                y_prob = None

            average = 'macro' if label_type == "Multiclass" else 'binary'
            acc = accuracy_score(y_test, y_test_pred)
            prec = precision_score(y_test, y_test_pred, average=average, zero_division=0)
            rec = recall_score(y_test, y_test_pred, average=average, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average=average, zero_division=0)

            if y_prob is not None:
                try:
                    if label_type == "Multiclass":
                        roc_val = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                    else:
                        if y_prob.shape[1] == 2:
                            roc_val = roc_auc_score(y_test, y_prob[:, 1])
                        else:
                            roc_val = np.nan
                except Exception:
                    roc_val = np.nan
            else:
                roc_val = np.nan

            cm = confusion_matrix(y_test, y_test_pred)
            final_summary[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc': roc_val,
                'confusion_matrix': cm,
                'y_test': y_test,
                'y_prob': y_prob
            }

            print(f"Final {name} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, ROC: {roc_val if not np.isnan(roc_val) else 'nan'}")
            plot_confusion_matrix(cm, class_labels=np.unique(y_test), title=f"{name} (Final Test)")

            if y_prob is not None and label_type == "Binary":
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plot_roc_curve(fpr, tpr, roc_auc, name, label_type)
                except Exception:
                    pass

        return final_summary

# ------------------ Run as script ------------------
if __name__ == "__main__":
    # Replace these with your paths
    train_csv = "UNSW_NB15_training-set.csv"
    test_csv = "UNSW_NB15_testing-set.csv"

    pipeline = IntrusionDetectionPipeline(random_state=42, n_splits=5, n_pca=20,
                                          n_clusters_kmeans=8, n_components_gmm=8,
                                          use_minibatch=True)
    pipeline.load_data(train_csv, test_csv)
    results = pipeline.run()

    # Example: print Decision Forest final metrics
    if results and results['final_binary']:
        for mdl_name, stats in results['final_binary'].items():
            print(f"\nMODEL: {mdl_name}")
            print(f"  Accuracy: {stats['accuracy']:.4f}")
            print(f"  Precision: {stats['precision']:.4f}")
            print(f"  Recall: {stats['recall']:.4f}")
            print(f"  F1: {stats['f1']:.4f}")
            print(f"  ROC: {stats['roc']}")
