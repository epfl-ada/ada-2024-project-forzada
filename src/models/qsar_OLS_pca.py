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
from src.models.helper_functions_qsar import *

# Define reusable preprocessing tools
scaler_cdk = StandardScaler()
pca = PCA(n_components=0.95)

# What you can do with this script:
# 1. create a model for a specific CDK
# 2. predict the log(IC50) for a specific molecule
# 3. compare the model with a shuffled model
# 4. plot the variance explained by PCA components

# TODO: set PCA as default in helper function to create features and target arrays


def create_model(df: pd.DataFrame, cdk, random_state=42, col="Target Name"):
    df = preprocess_df(df, cdk, col=col)
    X_train, X_test, y_train, y_test, scaler, pca = make_features_and_target_PCA(
        df, state=random_state
    )
    model = train_model(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, scaler, pca


def plot_variance_explained(X):
    """
    Plots the cumulative variance explained by PCA components.
    Use if PCA was used during training.
    """
    X_scaled = scaler_cdk.fit_transform(X)
    pca_full = PCA().fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Variance Explained")
    plt.grid()
    plt.show()


def print_model_summary(model, model_summary=False):
    if model_summary == True:
        print(model.summary())
    print("-----------------")
    print("R2: ", model.rsquared)
    print("MSE: ", model.mse_resid)
    print("RMSE: ", np.sqrt(model.mse_resid))


def compute_average_r2_rmse(df, cdk, random_states, col="Target Name"):
    """
    Compute the average of r2 and rmse for each of the models created with different random states.

    Args:
        df,
        cdk,
        random_states list

    Returns:
        r2_s (list),
        average of r2_s (float),
        rmse_s (list),
        average of rmse_s (float)
    """
    r2_s = []
    rmse_s = []
    for i in range(len(random_states)):
        # Make one model
        model, _, _, _, _, _, _ = create_model(
            df, [cdk], random_state=random_states[i], col=col
        )

        # Save the r2 and the rmse
        r2_s.append(model.rsquared)
        rmse_s.append(np.sqrt(model.mse_resid))

    return r2_s, np.mean(r2_s), rmse_s, np.mean(rmse_s)


def predict_log_affinity(model, smiles: str, scaler=scaler_cdk, pca=pca):
    """
    Predicts log(IC50) for a given SMILES string.
    Specify whether to use PCA or Variance Threshold.

    Args:
        model: Trained OLS model.
        smiles: SMILES string of the molecule.

    Returns:
        float: Predicted log(IC50) value.
    """
    # Convert SMILES to a fingerprint
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string.")
    fingerprint = mol2fp(mol).reshape(1, -1)

    fp_pca = pca.transform(fingerprint)
    fp_scaled = scaler.transform(fp_pca)
    fp_with_const = np.insert(fp_scaled, 0, 1, axis=1)

    # Ensure the dimensions match the model
    if fp_with_const.shape[1] != len(model.params):
        raise ValueError(
            f"Feature dimension mismatch: model expects {len(model.params)}, got {fp_with_const.shape[1]}."
        )

    # Make prediction
    return model.predict(fp_with_const)


def compare_cdkmodel_and_shuffled(model, y_train, X_train, y_test, X_test):
    """Constructs a new random model that trains a model based on the same set of features but with shuffled target values"""

    sns.set_theme(style="whitegrid")
    y_shuffled = np.random.permutation(y_train)
    model_shuffled = train_model(X_train, y_shuffled)

    # Compare R2 and RMSE of the two models
    print("R2 of the original model: ", model.rsquared)
    print("R2 of the shuffled model: ", model_shuffled.rsquared)
    print("RMSE of the original model: ", np.sqrt(model.mse_resid))
    print("RMSE of the shuffled model: ", np.sqrt(model_shuffled.mse_resid))

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_shuffled = model_shuffled.predict(X_test)

    # Determine consistent axis limits with padding
    def calculate_limits(data1, data2, padding=0.05):
        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        range_val = max_val - min_val
        return min_val - padding * range_val, max_val + padding * range_val

    residual_limits = calculate_limits(y_test - y_pred, y_test - y_pred_shuffled)
    prediction_limits = calculate_limits(
        np.concatenate([y_test, y_pred, y_pred_shuffled]),
        np.concatenate([y_test, y_pred, y_pred_shuffled]),
    )

    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Shared color palette
    color_original = sns.color_palette("muted")[0]
    color_shuffled = sns.color_palette("muted")[1]

    # Top-left: Residuals of the original model
    axes[0, 0].scatter(
        y_pred, y_test - y_pred, alpha=0.7, color=color_original, label="Residuals"
    )
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals: Original Model", fontsize=12, weight="bold")
    axes[0, 0].set_xlabel("Predicted Values", fontsize=10)
    axes[0, 0].set_ylabel("Residuals", fontsize=10)
    axes[0, 0].set_xlim(prediction_limits)
    axes[0, 0].set_ylim(residual_limits)

    # Top-right: Residuals of the shuffled model
    axes[0, 1].scatter(
        y_pred_shuffled,
        y_test - y_pred_shuffled,
        alpha=0.7,
        color=color_shuffled,
        label="Residuals",
    )
    axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Residuals: Shuffled Model", fontsize=12, weight="bold")
    axes[0, 1].set_xlabel("Predicted Values", fontsize=10)
    axes[0, 1].set_ylabel("Residuals", fontsize=10)
    axes[0, 1].set_xlim(prediction_limits)
    axes[0, 1].set_ylim(residual_limits)

    # Bottom-left: True vs Predicted (original)
    axes[1, 0].scatter(
        y_test, y_pred, alpha=0.7, color=color_original, label="True vs Predicted"
    )
    axes[1, 0].plot(
        prediction_limits, prediction_limits, color="red", linestyle="--", linewidth=1
    )
    axes[1, 0].set_title(
        "True vs Predicted: Original Model", fontsize=12, weight="bold"
    )
    axes[1, 0].set_xlabel("True Values", fontsize=10)
    axes[1, 0].set_ylabel("Predicted Values", fontsize=10)
    axes[1, 0].set_xlim(prediction_limits)
    axes[1, 0].set_ylim(prediction_limits)

    # Bottom-right: True vs Predicted (shuffled)
    axes[1, 1].scatter(
        y_test,
        y_pred_shuffled,
        alpha=0.7,
        color=color_shuffled,
        label="True vs Predicted",
    )
    axes[1, 1].plot(
        prediction_limits, prediction_limits, color="red", linestyle="--", linewidth=1
    )
    axes[1, 1].set_title(
        "True vs Predicted: Shuffled Model", fontsize=12, weight="bold"
    )
    axes[1, 1].set_xlabel("True Values", fontsize=10)
    axes[1, 1].set_ylabel("Predicted Values", fontsize=10)
    axes[1, 1].set_xlim(prediction_limits)
    axes[1, 1].set_ylim(prediction_limits)

    # Add annotations
    fig.suptitle(
        "Comparison of Original and Shuffled Models", fontsize=14, weight="bold"
    )

    # Add legend to differentiate plots
    for ax in axes.flat:
        ax.legend(loc="upper left", fontsize=9)

    plt.show()


if __name__ == "__main__":
    # Load data and choose CDKs
    df = pd.read_csv("../data/CDK_cleaned_for_families_prediction.csv")
    cdks = [
        "CDK5",
        "CDK1-G2/M-Cyc1",
        "CDK1",
        "CDK1-CycA2",
        "CDK1-G2/M-CycB",
        "CDK3",
        "CDK2[A144G]",
        "CDK2[F80T]",
        "CDK2[C118L,A144C]",
        "CDK2[A144C]",
        "CDK2",
        "CDK2-CycA2[171-432]",
        "CDK2-G1/S-CycE1",
        "CDK2-CycA2",
        "CDK2-CycA2[177-432]",
        "CDK2[F80M]",
        "CDK2-G1/S-CycE1-GSTP",
        "CDK2-CycA1",
        "CDK2[C118L]",
        "CDK2[C118I]",
    ]

    # Feature engineering with PCA

    model, X_train, X_test, y_train, y_test, scaler, pca = create_model(
        df, cdks, col="Cleaned Target Name"
    )

    print(
        f"Number of features after PCA: {X_train.shape[1] - 1}"
    )  # Subtract 1 for the constant term

    # compare_cdkmodel_and_shuffled(model, y_train, X_train, y_test, X_test)

    # # Model training
    # model = train_model(X_train, y_train)

    # # Evaluation
    # print_model_summary(model, model_summary=False)
    # plot_model_evaluation(model, X_test, y_test)

    # # Prediction example
    # test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    # log_ic50 = predict_log_affinity(model, test_smiles, var_thres=False)
    # print(f"Predicted log(IC50): {log_ic50[0]}")
    # print(f"Predicted IC50 (nM): {np.exp(log_ic50[0])}")

    # compare_cdkmodel_and_shuffled(model, y_train, X_train, y_test, X_test)
    pred = predict_log_affinity(
        model,
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        scaler=scaler,
        pca=pca,
    )

    print(f"prediction of smiles molecule is: {pred}")
