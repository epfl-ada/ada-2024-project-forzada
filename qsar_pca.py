import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

feature_select_cdk = VarianceThreshold(threshold=0.05)
scaler_cdk = StandardScaler()
pca_cdk = PCA()

feature_select_rand = VarianceThreshold(threshold=0.05)
scaler_rand = StandardScaler()


def mol2fp(mol):
    """Converts a molecule to a fingerprint"""
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=4096)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar


def preprocess_df(df: pd.DataFrame, cdks: list[str], col="Target Name") -> pd.DataFrame:
    """Takes an already cleaned dataframe of IC50 values and adds fingerprints of each smiles and add the log values of each IC50

    Args:
        df (pd.DataFrame): Dataframe of IC50 values. A valid df is one i.e. used for Project Milestone 2 describing IC50 values of ligand smiles and CDK. Ideally you have at this already selected with cdk you wish to screen for.

    Returns:
        pd.DataFrame: A Dataframe with added fingerprint for each smiles.
    """

    # filtering the df down to only contains rows with desired cdks
    df = df[df[col].isin(cdks)]

    # adding log values of IC50
    df["log_IC50"] = np.log(df["IC50 (nM)"].values)
    # dropping values with nan values for IC50
    df = df.dropna(subset=["log_IC50"])
    # resetting index
    df = df.reset_index(drop=True)

    # Creating a Mol column
    PandasTools.AddMoleculeColumnToFrame(df, "Ligand SMILES", "Molecule")
    # Vectorising the smiles
    df["Fingerprint"] = df["Molecule"].apply(mol2fp)

    return df


def remove_highly_correlated_features(X, threshold=0.85):
    """Removes highly correlated features based on a correlation threshold."""
    corr_matrix = pd.DataFrame(X).corr().abs()
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [col for col in range(corr_matrix.shape[1]) if any(corr_matrix.iloc[:, col][upper_triangle[:, col]] > threshold)]
    return np.delete(X, to_drop, axis=1)


def calculate_vif(X):
    """Calculates Variance Inflation Factor (VIF) for all features."""
    vif = pd.DataFrame()
    vif["Feature"] = range(X.shape[1])
    vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif


def make_features_and_target(
    df: pd.DataFrame, scaler=scaler_cdk, feature_select=feature_select_cdk, remove_multicollinearity=False
) -> tuple:
    """Takes a dataframe with fingerprints and log values of IC50 and returns a tuple of features and target for test and train set

    Args:
        df (pd.DataFrame): A dataframe with fingerprints and log values of IC50
        remove_multicollinearity (bool, optional): Whether to remove multicollinearity. Defaults to False.

    Returns:
        test / train features and target

    """
    
    # Splitting the data
    X = np.stack(df["Fingerprint"].values)
    y = df["log_IC50"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature selection
    X_train = feature_select.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = feature_select.transform(X_test)
    X_test = scaler.transform(X_test)

    if remove_multicollinearity:
        X_train = remove_highly_correlated_features(X_train)
        X_test = remove_highly_correlated_features(X_test)

    # Scaling
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    return X_train, X_test, y_train, y_test


### PCA for feature selection ###


def make_features_and_target_PCA(
    df: pd.DataFrame, scaler=scaler_cdk, feature_select=feature_select_cdk, pca=pca_cdk
) -> tuple:
    '''
    Applies dimensionality reduction using PCA.
    '''

    # Figure out how many components to keep 0.80 variance
    X = np.stack(df["Fingerprint"].values)
    y = df["log_IC50"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = feature_select.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_pca = pca.fit_transform(X_train_scaled)

    X_test = feature_select.transform(X_test)
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    # var_ratio = []
    # nums = np.arange(1, X_train.shape[1])
    # for num in nums:
    #     pca = PCA(n_components=num)
    #     pca.fit(X_train_scaled)
    #     var_ratio.append(np.sum(pca.explained_variance_ratio_))

    # plt.figure(figsize=(4, 2), dpi=150)
    # plt.grid()
    # plt.plot(nums, var_ratio, marker="o")
    # plt.xlabel("n_components")
    # plt.ylabel("Explained variance ratio")
    # plt.title("n_components vs. Explained Variance Ratio")
    # plt.show()

    return X_train_pca, X_test_pca, y_train, y_test


def train_model(X_train, y_train):
    """Trains an OLS regression model."""

    model = sm.OLS(y_train, X_train)
    res = model.fit()
    print(res.summary())
    return res


def evaluate_model(model, X_test, y_test):
    """Evaluates the model using R2, MSE, and MAE."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test) if hasattr(model, "score") else sm.OLS(y_test, X_test).fit().rsquared
    print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")


def plot_evaluation_of_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # Creating 2 plots for the evaluation of the model
    # 1 plotting the residuals of the model
    # 2 plotting the predicted vs true values

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(y_pred, y_test - y_pred, alpha=0.5)
    ax[0].set_xlabel("Predicted Values")
    ax[0].set_ylabel("Residuals")
    ax[0].set_title("Residuals of the model")

    ax[1].scatter(y_test, y_pred, alpha=0.5)
    ax[1].plot(
        [min(y_test), max(y_test)], [min(y_test), max(y_test)], ls="--", c="red", lw=4
    )
    ax[1].set_xlabel("True Values")
    ax[1].set_ylabel("Predicted Values")
    ax[1].set_title("Predicted vs True values")

    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/IC50_df.csv")
    cdks = [
        "Cyclin-dependent kinase 2/G1/S-specific cyclin-E1",
        "Cyclin-A2/Cyclin-dependent kinase 2",
    ]
    df_filtered = preprocess_df(df, cdks)

    # Multicollinearity-aware feature creation
    X_train, X_test, y_train, y_test = make_features_and_target(df_filtered)

    # Train OLS model
    ols_model = train_model(X_train, y_train)

    # Evaluate OLS model
    evaluate_model(ols_model, X_test, y_test)

    # Plot evaluation of OLS model
    plot_evaluation_of_model(ols_model, X_test, y_test)

    # # Train regularized models
    # ridge_model = train_regularized_model(X_train, y_train, model_type="ridge", alpha=1.0)
    # lasso_model = train_regularized_model(X_train, y_train, model_type="lasso", alpha=0.1)

    # # Evaluate models
    # evaluate_model(ridge_model, X_test, y_test)
    # evaluate_model(lasso_model, X_test, y_test)

    # # PCA-based model
    # X_train_pca, X_test_pca, y_train, y_test = make_features_and_target_PCA(df_filtered)
    # pca_model = train_model(X_train_pca, y_train)
