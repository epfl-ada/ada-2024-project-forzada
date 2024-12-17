import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Levenshtein import distance
from typing import List
from rdkit import Chem, DataStructs
import matplotlib.colors as mcolors

# Format the binding affinity value given
def clean(binding_value, pattern=['<', '>']):
    # we decided to simply remove the < and > signs assuming it woudn't impact the accuracy of our 
    # further analysis as we will use ranges of affinity (ex: strong, moderate, weak affinity)
    binding_value = str(binding_value).replace(pattern[0], '') 
    binding_value = str(binding_value).replace(pattern[1], '')
    
    binding_value = binding_value.strip() # remove spaces
    try:
        binding_value = int(binding_value) # convert into integers
    except ValueError:  
        binding_value = None
    return (binding_value)

def clean_target_name(row):
    if row['Target Name'] == 'Cyclin-dependent kinase/G2/mitotic-specific cyclin- 1':
        return 'CDK1-G2/M-Cyc1'
    elif row['Target Name'] == 'Cyclin-dependent kinase 2':
        return 'CDK2'
    elif row['Target Name'] == 'Cyclin-dependent kinase 1':
        return 'CDK1'
    elif row['Target Name'] == 'Cyclin-dependent kinase 1/G2/mitotic-specific cyclin-B':
        return 'CDK1-G2/M-CycB'
    elif row['Target Name'] == 'Cyclin-A2 [171-432]/Cyclin-dependent kinase 2':
        return 'CDK2-CycA2[171-432]'
    elif row['Target Name'] == 'Cyclin-dependent kinase 4/G1/S-specific cyclin-D1':
        return 'CDK4-G1/S-CycD1'
    elif row['Target Name'] == 'Cyclin-dependent kinase 2/G1/S-specific cyclin-E1':
        return 'CDK2-G1/S-CycE1'
    elif row['Target Name'] == 'Cyclin-dependent kinase 4/G1/S-specific cyclin-D1 [L188C]':
        return 'CDK4-G1/S-CycD1[L188C]'
    elif row['Target Name'] == 'Cyclin-A2/Cyclin-dependent kinase 2':
        return 'CDK2-CycA2'
    elif row['Target Name'] == 'Cyclin-dependent kinase 6/G1/S-specific cyclin-D1 [L188C]':
        return 'CDK6-G1/S-CycD1[L188C]'
    elif row['Target Name'] == 'Cyclin-dependent kinase 5 activator 1 [99-307]':
        return 'CDK5-Act1[99-307]'
    elif row['Target Name'] == 'Cyclin-dependent kinase 5 activator 1':
        return 'CDK5-Act1'
    elif row['Target Name'] == 'Cyclin-dependent kinase 4':
        return 'CDK4'
    elif row['Target Name'] == 'Cyclin-H/Cyclin-dependent kinase 7':
        return 'CDK7-CycH'
    elif row['Target Name'] == 'Cyclin-T1/Cyclin-dependent kinase 9':
        return 'CDK9-CycT1'
    elif row['Target Name'] == 'Cyclin-A2 [171-432]/Cyclin-dependent kinase 2 [K89T]':
        return 'CDK2[K89T]-CycA2[171-432]'
    elif row['Target Name'] == 'Cyclin-A2 [171-432]/Cyclin-dependent kinase 2 [L83V,H84D]':
        return 'CDK2[L83V,H84D]-CycA2[171-432]'
    elif row['Target Name'] == 'Cyclin-A2 [171-432]/Cyclin-dependent kinase 2 [F82H]':
        return 'CDK2[F82H]-CycA2[171-432]'
    elif row['Target Name'] == 'Cyclin-A2 [171-432]/Cyclin-dependent kinase 2 [F82H,L83V,H84D]':
        return 'CDK2[F82H,L83V,H84D]-CycA2[171-432]'
    elif row['Target Name'] == 'Cyclin-dependent kinase 7':
        return 'CDK7'
    elif row['Target Name'] == 'Cyclin-dependent kinase 9':
        return 'CDK9'
    elif row['Target Name'] == 'Cyclin-A2 [177-432]/Cyclin-dependent kinase 2':
        return 'CDK2-CycA2[177-432]'
    elif row['Target Name'] == 'Cyclin-dependent kinase 3':
        return 'CDK3'
    elif row['Target Name'] == 'Cyclin-dependent kinase 5':
        return 'CDK5'
    elif row['Target Name'] == 'Cyclin-dependent kinase 6/G1/S-specific cyclin-D3':
        return 'CDK6-G1/S-CycD3'
    elif row['Target Name'] == 'Cyclin-dependent kinase 3/G1/S-specific cyclin-E1':
        return 'CDK3-G1/S-CycE1'
    elif row['Target Name'] == 'Cyclin-dependent kinase 6':
        return 'CDK6'
    elif row['Target Name'] == 'Cyclin-dependent kinase 2/G1/S-specific cyclin-E1/Glutathione S-transferase P':
        return 'CDK2-G1/S-CycE1-GSTP'
    elif row['Target Name'] == 'Cyclin-A1/Cyclin-dependent kinase 2':
        return 'CDK2-CycA1'
    elif row['Target Name'] == 'Cyclin-A2 [171-432]/Cyclin-dependent kinase 2 [F82H,L83V,H84D,K98T]':
        return 'CDK2[F82H,L83V,H84D,K98T]-CycA2[171-432]'
    else:
        return row['Target Name']

# Classify the IC50 values 
def classify_IC50(value):
    thresholds = {
    'strong' : 100,
    'moderate' : 1000,
    'weak' : 10000,
    }
    if value < thresholds['strong']:
        return 'strong'
    elif value < thresholds['moderate']:
        return 'moderate'
    else:
        return 'weak'
    


def plot_cdk_confusion_matrix_non_normalized(df, sequence_column="BindingDB Target Chain Sequence", label_column="Target Name", plot=True):
    """
    Generates a confusion matrix plot of non-normalized Levenshtein distances for unique CDK sequences based on unique target names.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing CDK sequences and their labels.
    sequence_column (str): Column name in df that contains the CDK amino acid sequences.
    label_column (str): Column name in df that contains the labels for the CDK sequences (e.g., "Target Name").
    plot (bool): Whether to display the confusion matrix plot. Default is True.
    
    Returns:
    pd.DataFrame: Confusion matrix of Levenshtein distances.
    """
    # Step 1: Extract unique target names and corresponding sequences
    unique_df = df[[sequence_column, label_column]].dropna().drop_duplicates(subset=label_column)
    unique_sequences = [seq.lower() for seq in unique_df[sequence_column].tolist()]  # Convert to lowercase for case-insensitive comparison
    labels = unique_df[label_column].tolist()
    
    # Step 2: Initialize a DataFrame to store the Levenshtein distances
    distance_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    
    # Step 3: Calculate pairwise Levenshtein distances (without normalization)
    for i in range(len(unique_sequences)):
        for j in range(i, len(unique_sequences)):
            if i == j:
                distance_matrix.iloc[i, j] = 0
            else:
                dist = distance(unique_sequences[i], unique_sequences[j])
                distance_matrix.iloc[i, j] = dist
                distance_matrix.iloc[j, i] = dist  # Symmetric assignment

    # Step 4: Plot the confusion matrix if plot=True
    if plot:
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            distance_matrix,
            annot=False,  # Turn off annotations in each cell
            fmt=".0f",
            cmap="YlGnBu",
            square=True,
            cbar_kws={'label': 'Levenshtein Distance'},
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=90)
        plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)

        plt.title("Levenshtein Distance Matrix for Unique CDK Sequences (Non-Normalized, Based on Target Names)")
        plt.xlabel("CDK Target Names")
        plt.ylabel("CDK Target Names")
        plt.show()

    return distance_matrix

def plot_cdk_confusion_matrix_normalized(df, sequence_column="BindingDB Target Chain Sequence", label_column="Cleaned Target Name", plot=True):
    """
    Generates a confusion matrix plot of normalized Levenshtein distances for unique CDK sequences based on unique target names.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing CDK sequences and their labels.
    sequence_column (str): Column name in df that contains the CDK amino acid sequences.
    label_column (str): Column name in df that contains the labels for the CDK sequences (e.g., "Target Name").
    plot (bool): Whether to display the confusion matrix plot. Default is True.
    
    Returns:
    pd.DataFrame: Confusion matrix of Levenshtein distances.
    """
    unique_df = df[[sequence_column, label_column]].dropna().drop_duplicates(subset=label_column)
    unique_sequences = [seq.lower() for seq in unique_df[sequence_column].tolist()]  # Convert to lowercase for case-insensitive comparison
    labels = unique_df[label_column].tolist()
    
    distance_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    
    for i in range(len(unique_sequences)):
        for j in range(i, len(unique_sequences)):
            if i == j:
                distance_matrix.iloc[i, j] = 0
            else:
                dist = distance(unique_sequences[i], unique_sequences[j])
                max_len = max(len(unique_sequences[i]), len(unique_sequences[j]))
                normalized_dist = dist / max_len
                distance_matrix.iloc[i, j] = normalized_dist
                distance_matrix.iloc[j, i] = normalized_dist

    if plot:
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            distance_matrix,
            annot=False,
            fmt=".2f",
            cmap="YlGnBu",
            square=True,
            cbar_kws={'label': 'Levenshtein Distance'},
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=90)
        plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)

        plt.title("Levenshtein Distance Matrix for Unique CDK Sequences (Based on Target Names)")
        plt.xlabel("CDK Target Names")
        plt.ylabel("CDK Target Names")
        plt.show()

    return distance_matrix





def get_similar_cdks(distance_matrix, top_n=10):
    # Flatten the upper triangle of the distance matrix (excluding the diagonal) into a list of tuples
    pairs = []
    labels = distance_matrix.index.tolist()
    
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):  # Only consider upper triangle to avoid duplicates
            pairs.append((labels[i], labels[j], distance_matrix.iloc[i, j]))

    # Convert to DataFrame for easier sorting and selection
    pairs_df = pd.DataFrame(pairs, columns=["Second Amino Acid sequence (chain 1)", "Second Amino Acid sequence (chain 1)", "Distance"])

    # Sort by distance
    most_similar = pairs_df.nsmallest(top_n, "Distance")
    least_similar = pairs_df.nlargest(top_n, "Distance")
    
    return most_similar, least_similar



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


    # Get ligands info
    ligands_name_list = df["BindingDB Ligand Name"].tolist()
    #ligands_smiles_list = df["Ligand SMILES_x"].tolist()
    ligands_smiles_list = df["Ligand SMILES"].tolist()
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df,
        cmap="rocket",
        yticklabels=False, xticklabels=False,
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


    # coloring the values based on the threshold

    cmap = mcolors.ListedColormap(["#FF7A65", "#01F82B"])

    binary_matrix = np.where(similarity_matrix >= threshold, 1, 0)

    # Plot
    plt.figure(figsize=(8, 6))
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

    df1 = df[df[col] == val1]
    df2 = df[df[col] == val2]

    df_intersection = pd.merge(df1, df2, how="inner", on=merge_on)
    return df_intersection



def summary_of_ligand_similarities(similarity_matrix):
    if similarity_matrix.size == 0 or similarity_matrix.shape[0] == 0:

        return pd.Series({
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "no. similar ligands meas. >= 0.85": 0,
            "no. of non-similar ligands meas. < 0.85": 0,
            "count": 0,
            "similar ligands %": np.nan,
        })
    
    lower_triangle = np.tril(similarity_matrix).flatten()
    lower_triangle = lower_triangle[lower_triangle != 1]
    lower_triangle = lower_triangle[lower_triangle != 0]

    if lower_triangle.size == 0:
        return pd.Series({
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "no. similar ligands meas. >= 0.85": 0,
            "no. of non-similar ligands meas. < 0.85": 0,
            "count": 0,
            "similar ligands %": np.nan,
        })

    summary = {
        "mean": np.mean(lower_triangle),
        "std": np.std(lower_triangle),
        "min": np.min(lower_triangle),
        "max": np.max(lower_triangle),
        "no. similar ligands meas. >= 0.85": len(lower_triangle[lower_triangle >= 0.85]),
        "no. of non-similar ligands meas. < 0.85": len(lower_triangle[lower_triangle < 0.85]),
        "count": len(lower_triangle),
        "similar ligands %": len(lower_triangle[lower_triangle >= 0.85]) / len(lower_triangle) * 100,
    }

    return pd.Series(summary).T
