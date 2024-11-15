from typing import List
from rdkit import Chem, DataStructs


def compare_2_smiles(mol1, mol2) -> float:
    """Takes 2 SMILES strings as arguments and returns the Tanimoto similarity between the two molecules

    if return values > 0.85 they are considere similar

    """

    mol1 = Chem.MolFromSmiles(mol1)
    mol2 = Chem.MolFromSmiles(mol2)
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)

    T = DataStructs.TanimotoSimilarity(fp1, fp2)
    return T


def Tanimoto_list_of_smiles(smiles_list):
    """Takes a list of SMILES strings and returns a list of tuples with the Tanimoto similarity between each pair of molecules"""
    result = []

    for i in range(len(smiles_list)):
        for j in range(i + 1, len(smiles_list)):
            T = compare_2_smiles(smiles_list[i], smiles_list[j])
            result.append(T)

    result.sort(reverse=True)
    return result


def plot_tanimoto_heatmap(df):
    ligands_name_list = df["BindingDB Ligand Name"].tolist()
    ligands_smiles_list = df["Ligand SMILES"].tolist()

    similarity_matrix = pd.DataFrame(
        index=ligands_name_list, columns=ligands_name_list, dtype=float
    )

    for i in range(len(ligands_name_list)):
        for j in range(i, len(ligands_name_list)):
            m1 = Chem.MolFromSmiles(ligands_smiles_list[i])
            m2 = Chem.MolFromSmiles(ligands_smiles_list[j])
            fp1 = Chem.RDKFingerprint(m1)
            fp2 = Chem.RDKFingerprint(m2)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

            similarity_matrix.loc[ligands_name_list[i], ligands_name_list[j]] = (
                tanimoto_similarity
            )
            similarity_matrix.loc[ligands_name_list[j], ligands_name_list[i]] = (
                tanimoto_similarity
            )

    return similarity_matrix


def ligand_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a similarity matrix between ligands based on their SMILES strings

    Args:
        df (pd.DataFrame): The dataframe containing the ligands information

    Returns:
        _type_: A DataFrame with the Tanimoto similarity between ligands
    """
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    from rdkit import DataStructs

    # Get ligands info
    ligands_name_list = df["BindingDB Ligand Name"].tolist()
    ligands_smiles_list = df["Ligand SMILES_x"].tolist()

    # Bulk convert SMILES to molecules and generate fingerprints
    mols = [Chem.MolFromSmiles(s) for s in ligands_smiles_list]
    fps = [Chem.RDKFingerprint(m) for m in mols]

    # Initialize similarity matrix
    n = len(fps)
    similarity_matrix = np.zeros((n, n))

    # Bulk calculate similarities
    for i in range(n):
        # Calculate similarities for upper triangle
        sims = np.array(
            [DataStructs.FingerprintSimilarity(fps[i], fps[j]) for j in range(i, n)]
        )
        similarity_matrix[i, i:] = sims
        # Mirror to lower triangle
        similarity_matrix[i:, i] = sims

    # Convert to DataFrame with labels
    similarity_matrix = pd.DataFrame(
        similarity_matrix, index=ligands_name_list, columns=ligands_name_list
    )
    return similarity_matrix


def plot_tanimoto_similarity(df: pd.DataFrame):

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Plot
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        df,
        cmap="rocket",
        yticklabels=True,
        cbar_kws={"label": "Tanimoto similarity"},
        square=True,
    )

    plt.title("Tanimoto similarity between ligands")
    plt.xlabel("Ligands", size=12)
    plt.ylabel("Ligands", size=12)

    # plt.xticks(rotation=90, ha="right")
    # plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


def plot_conditional_tanimoto(similarity_matrix, threshold=0.85):

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    # coloring the values based on the threshold

    cmap = mcolors.ListedColormap(["#FF7A65", "#01F82B"])

    binary_matrix = np.where(similarity_matrix >= threshold, 1, 0)

    # Plot
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        binary_matrix,
        cmap=cmap,
        xticklabels=False,
        yticklabels=False,
        square=True,
        cbar=False,
    )

    plt.title("Tanimoto similarity between ligands, threshold = 0.85", size=14, pad=20)
    plt.xlabel("Ligands", size=12)
    plt.ylabel("Ligands", size=12)

    plt.xticks(rotation=90, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


def create_df_on_intersection_of_values(df, col, val1, val2, merge_on):
    """Creates the intersection of two dataframes based on the values of a column

    Args:
        df (pd.DataFrame): The dataframe to be filtered
        col (str): The column to be filtered i.e. Target Name
        val1 (str): The first value to be filtered i.e. 'CDK1'
        val2 (str): The second value to be filtered i.e. 'CDK2'
        merge_on (str): The column to merge on i.e. 'BindingDB Ligand Name'

    Returns:
        Dataframe: The intersection of the two dataframes based on the values of the column
    """

    import pandas as pd

    df1 = df[df[col] == val1]
    df2 = df[df[col] == val2]

    df_intersection = pd.merge(df1, df2, how="inner", on=merge_on)
    return df_intersection


def classify_IC50(value):

    thresholds = {"strong": 100, "moderate": 1000, "weak": 10000}

    if value < thresholds["strong"]:
        return "strong"
    elif value < thresholds["moderate"]:
        return "moderate"
    else:
        return "weak"


def summary_of_ligand_similarities(similarity_matrix):

    import pandas as pd
    import numpy as np

    assert similarity_matrix.shape[0] == similarity_matrix.shape[1]

    lower_triangle = np.tril(similarity_matrix)

    # Flatten, remove ones in the diagonal and remove upper triangle

    lower_triangle = lower_triangle.flatten()
    lower_triangle = lower_triangle[lower_triangle != 1]
    lower_triangle = lower_triangle[lower_triangle != 0]

    # Summary

    summary = {
        "mean": np.mean(lower_triangle),
        "std": np.std(lower_triangle),
        "min": np.min(lower_triangle),
        "max": np.max(lower_triangle),
        "no. similar ligands meas. >= 0.85": len(
            lower_triangle[lower_triangle >= 0.85]
        ),
        "no. of non-similar ligands meas. < 0.85": len(
            lower_triangle[lower_triangle < 0.85]
        ),
        "count": len(lower_triangle),
        "similar ligands %": len(lower_triangle[lower_triangle >= 0.85])
        / len(lower_triangle)
        * 100,
    }

    return pd.Series(summary).T
