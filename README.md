# ada-2024-project-forzada

## "What makes a good inhibitor against cell proliferation?" 

### Abstract:
Cancer cells can grow quickly in the human body, but what allows them to grow, and more importantly can we inhibit this proliferation from happening? We know that cyclin-dependent kinases (CDKs) drive the cell cycle and can interact with diverse ligands. This project explores how CDKs interact with their ligands, studying their binding affinities, amino acid sequences as well as their ligand's structure. Through the analysis of comparing the CDKs and ligands based on their affinity scores, we confirmed using the Levenshtein distance that some CDKs within the same family have very similar chain1 amino acid sequences. However, for these specific families there was not sufficient data to compare the ligand and CDK pair interaction. Thus, CDKs were joined by their families (UniProt ID) to make meaningful comparisons on ligands structures (SMILES) with the Tanimoto Similarity measure. Ligands with a strong affinitity to multiple CDKs did on average have 15% similar looking molecular structures. 


### Research Questions we want to answer

- What are the similarities between CDK and ligands?
     - Does a group of ligands have strong binding affinities across multiple CDKs? This will reveal plausible good inhibitors.
     - Using a similarity measure such as Tanimoto, how much information are we disregarding by this method? 
- Are there similarities in the amino acid sequences that explain these similarities between CDKs and ligands?
- Are these similarities based on patterns in the amino acid sequences or ligand structures?
- (Future) Can we make a regression model (QSAR) to predict the potential binding affinity between CDK proteins and ligands (or for any random protein).
- (more to come)

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
- N-grams (not implemented yet). We plan to use this on the CDK sequences to potentially locate the binding site pattern that enables the inhibiting in the CDK-ligand pair. Using n-grams will allow us to look at sequences of characters in the amino acid sequences of the CDKs. We can experiment with different values. 

**Prediction methods**:

- Using the QSAR method based on Tanimoto Similarity Coefficient as a variable (not implemented yet).



### Proposed timeline

| Phase                   | Description                                      | Duration       |
|-------------------------|--------------------------------------------------|----------------|
| **Data Preprocessing**  | Filter, clean, and normalize CDK-ligand data     | Before P2      |
| **Visualization**       | Create heatmaps for affinity scores              | Before P2      |
| **Sequence Embedding**  | Encode sequences for similarity measurement      | (Week 1)       |
| **QSAR Modeling**       | Try to build regression model for affinity prediction   | Weeks 1-3      |
| **Final Analysis**      | Interpret results                                | Week 4         |
| **Hand-in**             | Finalize README and datastory                    | Week 5         |



### Organization within the team
Internal milestones for milestone P3: 
* **[Done]** Data exploration: clean data, handle outliers, normalise, drop redundant information
* **[Done]** Visualisation: visualise groups of CDK's and ligands to decide which deserve deeper exploration
* **[Partially done]**: Sequence Embedding: Using the Levenshtein distance and Tanimoto measure to compare the sequences of the CDKs and the ligand structures. We need to extend this to use N-grams so that we can find potential patterns that strongly inhibits binding affinity.
* Prediction method done if possible.


### Questions for TAs (optional)

How much data should we use? We feel we need to upscale and use more of the data, but we limited ourselves to only use the CDKs for this milestone. We think it could be interesting to use all (or at least more) of the data e.g. to cluster based on how targets interact with ligands (we would expect to find the CDK families).

We are unsure of how we should build a prediction model and if this will work.
