# ada-2024-project-forzada

## Shutting Down the Cell Cycle: Potential of CDK Inhibitors in Cancer treatment

### Abstract:
Cancer cells can grow quickly in the human body, but what allows them to grow, and more importantly can we inhibit this proliferation from happening? We know that cyclin-dependent kinases (CDKs) drive the cell cycle and can interact with diverse ligands. This project explores how CDKs interact with their ligands, studying their binding affinities, amino acid sequences as well as their ligand's structure. Through the analysis of comparing the CDKs and ligands based on their affinity scores, we confirmed using the Levenshtein distance that some CDKs within the same family have very similar chain1 amino acid sequences. However, for these specific families there was not sufficient data to compare the ligand and CDK pair interaction. Thus, CDKs were joined by their families (UniProt ID) to make meaningful comparisons on ligands structures (SMILES) with the Tanimoto Similarity measure. Ligands with a strong affinitity to multiple CDKs did on average have 15% similar looking molecular structures. 

Please check out our datastory [here](https://epfl-ada.github.io/ada-2024-project-forzada/)!

(url: https://epfl-ada.github.io/ada-2024-project-forzada/)

### Research Questions we want to answer

- What are the similarities between CDK and ligands?
     - Does a group of ligands have strong binding affinities across multiple CDKs? This will reveal plausible good inhibitors.
     - Using a similarity measure such as Tanimoto, how much information are we disregarding by this method? 
- Are there similarities in the amino acid sequences that explain these similarities between CDKs and ligands?
- Are these similarities based on patterns in the amino acid sequences or ligand structures?
- Can we make a regression model (QSAR) to predict the potential binding affinity between CDK proteins and ligands (or for any random protein).

In general, the subset containing only CDK's is not sufficient data and we expect to have to expand these method onto multiple target families.

### Proposed additional dataset
* PDP viewer as a potential additional dataset for P3. *We don't yet use this dataset, because we think it might be a very manual process and maybe too much focusing on a life science approach and not a data analysist approach.*
  
### Methods

**String matching methods**:

To gain insight into the similarities and differences between various CDK amino acid sequences, we will use string-based similarity measures. We plan to use
- Levenshtein distance measure. Which matches and measures the characters that are different in two given sequences. We can explain this in a confusion matrix of the unique CDKs and gain insight on similar CDKs.

**Pattern matching methods**:

To deepen our analysis we also want to utilize pattern matching methods to explore patterns in the CDK sequences and Ligand structures. 
- Tanimoto (for the SMILES). We use this for the ligand structures to compare the similarities. 
- Utilizing the rd2kit to find other measures for the molecules
- Morgan Fingerprint of the SMILES

**Prediction methods**:

Using the QSAR method based on the SMILES of the ligands as input and then predicting the affinity with a given CDK. We will use the Morgan Fingerprint for the SMILES and then using PCA to reduce the dimensionality. Afterwards we will train a linear (Ordinary Least Squares) model.
We will use this model to predict affinity scores of other SMILES, once that have not been tested with given CDKs.



### Proposed timeline

| Phase                   | Description                                      | Duration       |
|-------------------------|--------------------------------------------------|----------------|
| **Data Preprocessing**  | Filter, clean, and normalize CDK-ligand data     | Before P2      |
| **Visualization**       | Create heatmaps for affinity scores              | Before P2      |
| **Sequence Embedding**  | Encode sequences for similarity measurement      | (Week 1)       |
| **QSAR Modeling**       | Build regression model for affinity prediction   | Weeks 1-3      |
| **Final Analysis**      | Interpret results                                | Week 4         |
| **Hand-in**             | Finalize README and datastory, video and more    | Week 5         |



### Organization within the team
Internal milestones for milestone P3: 
* Data exploration: clean data, handle outliers, normalise, drop redundant information
* Visualisation: visualise groups of CDK's and ligands to decide which deserve deeper exploration
* Sequence Embedding: Using the Levenshtein distance and Tanimoto measure to compare the sequences of the CDKs and the ligand structures. We need to extend this to use N-grams so that we can find potential patterns that strongly inhibits binding affinity.
* Prediction method QSAR: Using the SMILES of the ligand molecules we predict the affinity scores (IC50) with a CDK or a CDK family. 


### Group member contribution
Everyone helped with writing the datastory and ensuring the reposity was cleaned.

| Member          | Main focus areas                                   |
|-----------------|----------------------------------------------------|
|Johanne          | QSAR model, similarity measures, final predictions |
|Julie            | Datastory writing, video, CDK families             |
|Mathias          | QSAR model, Tanimoto, final predictions            |
|Mathilde         | Preprocessing of data, datastory writing, video,   |
|Timon            | Fixing website, datastory                          |     
