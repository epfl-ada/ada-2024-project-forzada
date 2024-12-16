import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, PandasTools
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


feature_selector = VarianceThreshold(threshold=0.05)
scaler_cdk = StandardScaler()
pca = PCA(n_components=0.95)


def mol2fp(mol):
    """
    Converts a molecule to a fingerprint.
    Uses Morgan fingerprints with radius 2 and 4096 bits.
    Source: Esben Bjerrum's blog post
    https://www.cheminformania.com/building-a-simple-qsar-model-using-a-feed-forward-neural-network-in-pytorch/
    """
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=4096)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar


def preprocess_df(df: pd.DataFrame, cdks: list[str], col="Target Name") -> pd.DataFrame:
    """
    Takes a cleaned dataframe of IC50 values and adds fingerprints of each smiles and add the log values of each IC50

    Args:
        df (pd.DataFrame): Dataframe of IC50 values. A valid df is one i.e. used for Project Milestone 2 describing IC50 values of ligand smiles and CDK.
        Ideally you have at this already selected with cdk you wish to screen for.

    Returns:
        pd.DataFrame: A Dataframe with added fingerprint for each smiles.
    """

    # Filter the DataFrame to only contain rows with desired CDKs and create an explicit copy
    df = df[df[col].isin(cdks)].copy()

    # If a ligand smiles has been tested on multiple CDKs, keep an average of the IC50 values measured for this specific ligand
    # and drop the duplicates

    if len(cdks) > 1:

        df = df.groupby(["Ligand SMILES"], as_index=False).agg(
            {"IC50 (nM)": "mean", col: "first"}
        )
        df = df.drop_duplicates(subset=["Ligand SMILES"])

    # Adding log values of IC50 and dropping rows with NaN values
    df.loc[:, "log_IC50"] = np.log(df["IC50 (nM)"].values)
    df = df.dropna(subset=["log_IC50"])
    # Resetting index
    df = df.reset_index(drop=True)

    # Creating a column for the SMILES
    PandasTools.AddMoleculeColumnToFrame(df, "Ligand SMILES", "Molecule")
    # Vectorising
    df["Fingerprint"] = df["Molecule"].apply(mol2fp)

    return df


def make_features_and_target_var(df: pd.DataFrame, scaler=scaler_cdk) -> tuple:
    """
    Generates train/test features and target arrays, using variance threshold for dimensionality reduction.
    Args:
        df (pd.DataFrame): Preprocessed containing the fingerprints

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Splitting the data
    X = np.stack(df["Fingerprint"].values)
    y = df["log_IC50"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature selection
    X_train = feature_selector.fit_transform(X_train)
    X_test = feature_selector.transform(X_test)

    # Scaling the data
    X_train = scaler_cdk.fit_transform(X_train)
    X_test = scaler_cdk.transform(X_test)

    # Adding constant term for OLS
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a linear regression model using OLS.
    """
    model = sm.OLS(y_train, X_train).fit()
    return model


def plot_model_evaluation(model, X_test, y_test):
    """
    Plots the residuals and plots predicted vs true values for model evaluation.
    """
    y_pred = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(y_pred, y_test - y_pred, alpha=0.5)
    axes[0].set_title("Residuals of the model")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residuals")

    axes[1].scatter(y_test, y_pred, alpha=0.5)
    axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
    axes[1].set_title("True vs Predicted")
    axes[1].set_xlabel("True Values")
    axes[1].set_ylabel("Predicted Values")

    plt.tight_layout()
    plt.show()


def make_features_and_target_PCA(
    df: pd.DataFrame, scaler=scaler_cdk, state=42
) -> tuple:
    """
    Generates train/test features and target arrays, with PCA for dimensionality reduction.
    Args:
        df (pd.DataFrame): Preprocessed containing the fingerprints

    Returns:
        X_train, X_test, y_train, y_test, scaler, pca
    """
    # Splitting the data
    print(df["Fingerprint"][1])
    X = np.stack(df["Fingerprint"].values)
    y = df["log_IC50"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=state
    )

    # Applying PCA
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Scaling the data
    X_train = scaler_cdk.fit_transform(X_train)
    X_test = scaler_cdk.transform(X_test)

    # Adding constant term for OLS
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    return X_train, X_test, y_train, y_test, scaler, pca
