import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


feature_select_cdk = VarianceThreshold(threshold=0.05)
scaler_cdk = StandardScaler()

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


def make_features_and_target(
    df: pd.DataFrame, scaler=scaler_cdk, feature_select=feature_select_cdk
) -> tuple:
    """Takes a dataframe with fingerprints and log values of IC50 and returns a tuple of features and target for test and train set

    Args:
        df (pd.DataFrame): A dataframe with fingerprints and log values of IC50

    Returns:
        test / train features and target
    """

    # Splitting the data
    X = np.stack(df["Fingerprint"].values)
    y = df["log_IC50"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # feature selection

    X_train = feature_select.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_test = feature_select.transform(X_test)
    X_test = scaler.transform(X_test)

    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):

    model = sm.OLS(y_train, X_train)
    res = model.fit()
    print(res.summary())

    return res


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


def print_model_summary(model):
    print(model.summary())
    print("-----------------")
    print("R2: ", model.rsquared)


def create_me_a_model(df: pd.DataFrame, cdks: list[str]):
    df_filtered = preprocess_df(df, cdks)
    X_train, X_test, y_train, y_test = make_features_and_target(df_filtered)
    model = train_model(X_train, y_train)
    return model


def predict_log_affinity(
    model, fingerprint, scaler=scaler_cdk, feature_select=feature_select_cdk
):

    # We need to preprocess the fingerprint
    mol = Chem.MolFromSmiles(fingerprint)
    arr = mol2fp(mol).reshape(1, -1)

    print(f"Shape of fingerprint: {len(arr)}")
    print(f"This is the fingerprint: {fingerprint}")
    print(f"Shape of a feature vector: {model.params.shape}")
    ### Quite sure we will have to add a constant to the fingerprint i.e. 1 at the head of the vector
    print(f"Shape of fingerprint: {arr.shape}")

    row_select = feature_select.transform(arr)
    print(f"length of row_select: {len(row_select)}")
    row_scaled = scaler.transform(row_select)
    row_scaled = np.insert(row_scaled, 0, 1)

    return model.predict(row_scaled)


####################################
####################################
####################################
####################################
###### FUNCTIONS FOR TESTING #######
###### FUNCTIONS FOR TESTING #######
###### FUNCTIONS FOR TESTING #######
###### FUNCTIONS FOR TESTING #######
###### FUNCTIONS FOR TESTING #######
###### FUNCTIONS FOR TESTING #######
###### FUNCTIONS FOR TESTING #######
####################################
####################################
####################################
####################################



# Uniform random predictions
def generate_uniform_random_predictions(y_train, num_samples):
    """
    Generate random predictions uniformly within the range of the target variable.
    """
    y_min = y_train.min()
    y_max = y_train.max()
    return np.random.uniform(y_min, y_max, size=num_samples)


# Distribution-based random predictions
def generate_distribution_random_predictions(y_train, num_samples):
    """
    Generate random predictions based on the target variable's distribution.
    """
    mean = y_train.mean()
    std = y_train.std()
    return np.random.normal(loc=mean, scale=std, size=num_samples)


def evaluate_random_models(y_train, y_test):
    """
    Evaluate two random models (uniform and distribution-based) and compare their performance.
    """
    num_samples = len(y_test)

    # Uniform random predictions
    y_pred_uniform = generate_uniform_random_predictions(y_train, num_samples)
    mse_uniform = mean_squared_error(y_test, y_pred_uniform)
    r2_uniform = r2_score(y_test, y_pred_uniform)
    
    print(f"Uniform Random Model Performance:")
    print(f"Mean Squared Error: {mse_uniform:.4f}")
    print(f"R² Score: {r2_uniform:.4f}\n")

    # Distribution-based random predictions
    y_pred_distribution = generate_distribution_random_predictions(y_train, num_samples)
    mse_distribution = mean_squared_error(y_test, y_pred_distribution)
    r2_distribution = r2_score(y_test, y_pred_distribution)
    
    print(f"Distribution-Based Random Model Performance:")
    print(f"Mean Squared Error: {mse_distribution:.4f}")
    print(f"R² Score: {r2_distribution:.4f}\n")

    return {
        "uniform": {
            "predictions": y_pred_uniform,
            "mse": mse_uniform,
            "r2": r2_uniform,
        },
        "distribution": {
            "predictions": y_pred_distribution,
            "mse": mse_distribution,
            "r2": r2_distribution,
        }
    }


if __name__ == "__main__":
    # Assuming `df` is your dataset and you've already preprocessed it
    df = pd.read_csv("data/IC50_df.csv")
    cdks = [
        "Cyclin-dependent kinase 2/G1/S-specific cyclin-E1",
        "Cyclin-A2/Cyclin-dependent kinase 2",
    ]

    # Preprocess and split dataset
    df_filtered = preprocess_df(df, cdks)
    X_train, X_test, y_train, y_test = make_features_and_target(df_filtered)

    # Train your actual regression model
    model = train_model(X_train, y_train)
    print_model_summary(model)
    plot_evaluation_of_model(model, X_test, y_test)

    # Evaluate random models
    random_model_results = evaluate_random_models(y_train, y_test)

    # Compare random models with the regression model
    print("Actual Regression Model:")
    mse_linear = mean_squared_error(y_test, model.predict(X_test))
    r2_linear = r2_score(y_test, model.predict(X_test))
    print(f"Mean Squared Error: {mse_linear:.4f}")
    print(f"R² Score: {r2_linear:.4f}\n")








# def construct_random_IC50_df(df: pd.DataFrame, cdks) -> pd.DataFrame:

#     # Give new random IC50 values to the dataframe in the range of 1-100000 with a long tail towards the higher values
#     df["IC50 (nM)"] = np.random.exponential(10000, len(df))

#     df = preprocess_df(df, cdks)

#     return df






'''
def comparison_2_models(model1, random_model):
    print(f"model based on desired cdk: {model1.summary()}")
    print(
        f"model based on random IC50 values for the same selected cdk: {random_model.summary()}"
    )

    print(f"R2 for model based on desired cdk: {model1.rsquared}")
    print(
        f"R2 for model based on random IC50 values for the same selected cdk: {random_model.rsquared}"
    )


if __name__ == "__main__":
    # Testing the function
    df = pd.read_csv("data/IC50_df.csv")
    cdks = [
        "Cyclin-dependent kinase 2/G1/S-specific cyclin-E1",
        "Cyclin-A2/Cyclin-dependent kinase 2",
    ]
    df_filtered = preprocess_df(df, cdks)

    X_train, X_test, y_train, y_test = make_features_and_target(df_filtered)

    model = train_model(X_train, y_train)

    plot_evaluation_of_model(model, X_test, y_test)

    print_model_summary(model)

    # Testing the prediction
    # An arbitrary smiles string
    smiles = "COc1cc(CS(C)(=O)=NC#N)cc(Nc2ncc(F)c(n2)-c2ccc(F)cc2OC)c1"
    predicted_log_val = predict_log_affinity(model, smiles)
    print(f"Predicted IC50 log value for {smiles}: {predicted_log_val}")
    pred_val = np.exp(predicted_log_val)
    print(f"Predicted IC50 value for {smiles}: {pred_val}")

    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    aspirin_pred = predict_log_affinity(model, aspirin_smiles)
    print(f"Predicted IC50 log value for {aspirin_smiles}: {aspirin_pred}")
    aspirin_val = np.exp(aspirin_pred)
    print(f"Predicted IC50 value for {aspirin_smiles}: {aspirin_val}")
    print("Done")

    ethanol_smiles = "CCO"
    ethanol_pred = predict_log_affinity(model, ethanol_smiles)
    print(f"Predicted IC50 log value for {ethanol_smiles}: {ethanol_pred}")
    ethanol_val = np.exp(ethanol_pred)
    print(f"Predicted IC50 value for {ethanol_smiles}: {ethanol_val}")

    print("Done")

    # rand_df = construct_random_IC50_df(df, cdks)
    # X_train_rand, X_test_rand, y_train_rand, y_test_rand = make_features_and_target(
    #     rand_df, scaler=scaler_rand, feature_select=feature_select_rand
    # )
    # andom_model = train_model(X_train_rand, y_train_rand)

    # comparison_2_models(model, random_model)

    # # Testing the prediction on the random model on the same smiles
    # aspirin_log_val_rand = predict_log_affinity(random_model, aspirin_smiles, scaler=scaler_rand, feature_select=feature_select_rand)
    # print(f"Predicted IC50 log value for {smiles} with random model: {predicted_log_val_rand}")
    # pred_val_rand = np.exp(predicted_log_val_rand)
    # print(f"Predicted IC50 value for {smiles} with random model: {pred_val_rand}")
'''