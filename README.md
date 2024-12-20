# ada-2024-project-forzada

## Shutting Down the Cell Cycle: Potential of CDK Inhibitors in Cancer treatment 

### Abstract:

Globally, 1 in 6 people die from cancer. Despite their differences, all cancers share a common hallmark: uncontrolled cell proliferation. This cell division is governed by specific enzymes, the cyclin-dependent kinases (CDKs), activated when bound to specific cyclins. But in the case of cancers, the complex CDK/cyclin is overactivated ! In this project, our idea is to find a inhibitor that could target CDKs, prevent this anarchic cell division and maybe cure cancers ! 

To achieve this, we sort CDKs by families based on their amino acid sequence similarities. We explore how CDKs interact with their ligands, studying their binding affinities, amino acid sequences as well as their ligand's structure. From these analysis, we build a model to predict the affinity of a molecule with specific CDKs or families of CDKs. 

From this model, we are able to find potential inhibitors targeting a family of CDKs or a specific CDK. These inhibitors by interrupting the cell division, can stop an uncontrolled cell proliferation, and prevent the development of cancers! 

### Research Questions we want to answer

**Main question**
- Could we find potential inhibitors of CDKs?

**Derived questions**
- Do CDKs have similarities that could allow us to classify them into families?
- How do the CDKs interact with their ligands?
- Do CDKs of the same family have a similar interaction profile with their ligands?
- Do the ligands with high affinity for a CDK share common structural features?
- Can we design a model to predict the affinity of a given molecule with a specific CDK or family of CDKs? 


### Methods

**String matching methods**:

To gain insight into the similarities and differences between various CDK amino acid sequences, we will use string-based similarity measures. We plan to use
- Levenshtein distance measure. Which matches and measures the characters that are different in two given sequences. We can explain this in a confusion matrix of the unique CDKs and gain insight on similar CDKs.


**Pattern matching methods**:

To deepen our analysis we also want to utilize pattern matching methods to explore patterns in the CDK sequences and Ligand structures. 
- Tanimoto (for the SMILES). We use this for the ligand structures to compare the similarities.
- Morgan Fingerprint. Using the rd2kit we can obtain the Morgan fingerprint for a molecule. This is a way to vectorize the molecules so that we can feed it as input to a model.


**Prediction methods**:

- Using the QSAR method based on Tanimoto Similarity Coefficient as a variable.
- Using PCA to reduce dimensionality after using the Morgan Fingerprint to obtain a vectorized format of the molecule (SMILES)
- Evaluating using R2 scores and RMSE.
- Comparing the model against a random prediction model using shuffling technique. This will evaluate whether our model is better than a random model since to our knowledge there are no available, existing models made for CDK with ligands affinity scores.



### Proposed timeline

| Phase                   | Description                                      | Duration       |
|-------------------------|--------------------------------------------------|----------------|
| **Data Preprocessing**  | Filter, clean, and normalize CDK-ligand data     | Before P2      |
| **Visualization**       | Create heatmaps for affinity scores              | Before P2      |
| **Sequence Embedding**  | Encode sequences for similarity measurement      | (Week 1)       |
| **QSAR Modeling**       | Try to build regression model for affinity prediction   | Weeks 1-3      |
| **CDK families**        | Group the CDKs into families                     | Week 4         |
| **Final Analysis**      | Interpret results                                | Week 4         |
| **Hand-in**             | Finalize README and datastory + explanatory video    | Week 5         |



### Organization within the team
Internal milestones for milestone P3: 
* Data exploration: clean data, handle outliers, normalise, drop redundant information
* Visualisation: visualise groups of CDK's and ligands to decide which deserve deeper exploration
* Sequence Embedding: Using the Levenshtein distance and Tanimoto measure to compare the sequences of the CDKs and the ligand structures. Additionally use the Morgan Fingerprint
* Prediction method using QSAR. Dimensionality reduction using PCA.
* Analysing the output of the prediction

### Group member contributions

Everyone helped with writing the datastory and ensuring that the repository was up to date.


| Member     | Main tasks                                      |
|------------|-------------------------------------------------|
|Johanne     | QSAR model, similarity measures, final predictions|
|Julie       | Datastory writing, video, CDK family analysis   |
|Mathias     | QSAR, Tanimoto, final predictions               |
|Mathilde    | Preprocessing of data, datastory writing, video |
|Timon       | Website, datastory                              |

### Datastory website
https://epfl-ada.github.io/ada-2024-project-forzada/

