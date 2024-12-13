# Function to convert a dataframe into a FASTA folder
def dataframe_to_fasta(df, id_col, seq_col, output_file):
    """
    Parameters:
    df (pd.DataFrame): DataFramne containing sequences 
    id_col (str): name of the column containg name of the CDKs.
    seq_col (str): name of the column containing sequnces.
    output_file (str): fasta file.
    """
    with open(output_file, "w") as fasta_file:
        for _, row in df.iterrows():
            fasta_file.write(f">{row[id_col]}\n")  
            fasta_file.write(f"{row[seq_col]}\n") 

