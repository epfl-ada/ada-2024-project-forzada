import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define reusable preprocessing tools
#feature_selector = VarianceThreshold(threshold=0.05)
scaler = StandardScaler()
pca = PCA(n_components=0.75)

def mol2fp(mol):
    """Converts a molecule to a fingerprint"""
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=4096)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar

def preprocess_df(df: pd.DataFrame, cdks: list[str], col="Target Name") -> pd.DataFrame:
    """Filters and preprocesses the dataframe, adding molecular fingerprints and log IC50 values."""
    df = df[df[col].isin(cdks)].dropna(subset=["IC50 (nM)"]).reset_index(drop=True)
    df["log_IC50"] = np.log(df["IC50 (nM)"])
    PandasTools.AddMoleculeColumnToFrame(df, "Ligand SMILES", "Molecule")
    df["Fingerprint"] = df["Molecule"].apply(mol2fp)
    return df

def make_features_and_target(df: pd.DataFrame, use_pca=False) -> tuple:
    """Generates train/test features and target arrays, with optional PCA for dimensionality reduction."""
    X = np.stack(df["Fingerprint"].values)
    y = df["log_IC50"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature selection and scaling
    # X_train = feature_selector.fit_transform(X_train)
    # X_test = feature_selector.transform(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if use_pca:
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Adding constant term for OLS
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    return X_train, X_test, y_train, y_test

def plot_variance_explained(X):
    """Plots the cumulative variance explained by PCA components."""
    # X_scaled = scaler.fit_transform(feature_selector.fit_transform(X))
    X_scaled = scaler.fit_transform(X)
    pca_full = PCA().fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Variance Explained")
    plt.grid()
    plt.show()

def train_model(X_train, y_train):
    """Trains a linear regression model using OLS."""
    model = sm.OLS(y_train, X_train).fit()
    return model

def print_model_summary(model):
    # print(model.summary())
    print("-----------------")
    print("R2: ", model.rsquared)
    print("MSE: ", model.mse_resid)
    print("RMSE: ", np.sqrt(model.mse_resid))

def plot_model_evaluation(model, X_test, y_test):
    """Plots residuals and predicted vs true values for model evaluation."""
    y_pred = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(y_pred, y_test - y_pred, alpha=0.5)
    axes[0].set_title("Residuals vs Predicted")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residuals")

    axes[1].scatter(y_test, y_pred, alpha=0.5)
    axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
    axes[1].set_title("True vs Predicted")
    axes[1].set_xlabel("True Values")
    axes[1].set_ylabel("Predicted Values")

    plt.tight_layout()
    plt.show()

def predict_affinity(model, smiles: str, use_pca=False):
    """Predicts log(IC50) for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string.")
    
    # Convert the SMILES string to a fingerprint
    fingerprint = mol2fp(mol).reshape(1, -1)

    # Apply the same preprocessing steps as the training data
    #fingerprint = feature_selector.transform(fingerprint)
    fingerprint = scaler.transform(fingerprint)

    # Apply PCA only if it was used during training
    if use_pca:
        fingerprint = pca.transform(fingerprint)
    
    # Explicitly add the constant term to the fingerprint
    fingerprint_with_const = np.hstack([np.ones((fingerprint.shape[0], 1)), fingerprint])

    # Ensure the dimensions match the model
    if fingerprint_with_const.shape[1] != len(model.params):
        raise ValueError(
            f"Feature dimension mismatch: model expects {len(model.params)}, got {fingerprint_with_const.shape[1]}."
        )

    # Make prediction
    return model.predict(fingerprint_with_const)




if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/IC50_df.csv")
    cdks = ["Cyclin-dependent kinase 2", "Cyclin-A2/Cyclin-dependent kinase 2"]

    # Preprocessing
    df_preprocessed = preprocess_df(df, cdks)
    fingerprints = np.stack(df_preprocessed["Fingerprint"].values)
    print(f"Initial number of features (fingerprint size): {fingerprints.shape[1]}")

    # Apply Variance Threshold
    # fingerprints_selected = feature_selector.fit_transform(fingerprints)
    # print(f"Number of features after VarianceThreshold: {fingerprints_selected.shape[1]}")
    #df_preprocessed["Fingerprint_Selected"] = list(fingerprints_selected)
    
    # Inspecting the actual reduced features
    print("Sample features after VarianceThreshold:")
    # print(fingerprints_selected[0])  # Printing the first reduced fingerprint

    # Variance explained when using PCA
    plot_variance_explained(fingerprints)

    # Feature engineering with PCA
    X_train, X_test, y_train, y_test = make_features_and_target(df_preprocessed, use_pca=True)
    print(f"Number of features after PCA: {X_train.shape[1] - 1}")  # Subtract 1 for the constant term

    # Model training
    model = train_model(X_train, y_train)

    # print
    print_model_summary(model)

    # Evaluation
    plot_model_evaluation(model, X_test, y_test)

    # Prediction example
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    log_ic50 = predict_affinity(model, test_smiles, use_pca=True)
    print(f"Predicted log(IC50): {log_ic50[0]}")
    print(f"Predicted IC50 (nM): {np.exp(log_ic50[0])}")
