IC50_df = df[
    [
        "Ligand SMILES",
        "BindingDB Ligand Name",
        "Target Name",
        "Target Source Organism According to Curator or DataSource",
        "IC50 (nM)",
        "BindingDB Target Chain Sequence",
        "UniProt (SwissProt) Entry Name of Target Chain",
    ]
]

IC50_df.groupby(
    [
        "Ligand SMILES",
        "BindingDB Ligand Name",
        "Target Name",
        "Target Source Organism According to Curator or DataSource",
        "BindingDB Target Chain Sequence",
        "UniProt (SwissProt) Entry Name of Target Chain",
    ]
)["IC50 (nM)"].median().reset_index()

IC50_df["IC50_class"] = IC50_df["IC50 (nM)"].apply(
    lambda value: cf.classify_IC50(value)
)

IC50_ligand_strong = IC50_df[IC50_df["IC50_class"] == "strong"]
IC50_ligand_strong.groupby("BindingDB Ligand Name")

strong_ligands_cdks_2 = cf.create_df_on_intersection_of_values(
    IC50_ligand_strong,
    "Target Name",
    "Cyclin-A2/Cyclin-dependent kinase 2",
    "Cyclin-dependent kinase 2/G1/S-specific cyclin-E1",
    "BindingDB Ligand Name",
)
