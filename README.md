# ada-2024-project-forzada
ada-2024-project-forzada created by GitHub Classroom

Title: "Which Cyclic Dependent Kinase (CDK) has the best binding affinity?" (*to be edited*)

### Abstract: A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?

*150 words. to be written when the analysis is done*

 
### Additional dataset

*Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect.  Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.*



### Research Questions

A list of research questions you would like to address during the project.


1. Are the CDK interactions different depending on their nature (do CDK1 has higher affinities with its ligand than CDK2?) or their ligand (Do CDK1 interact better with its cyclins or with its inhibitors?)?
2. What characteristics are significant for the binding (patterns in the binding sequences, amino acid sequence on active site)? 
* Could we find CDK’s common features among the CDK family thanks to these clusters (families of ligands based on amino acid sequence, common ligands)?
* Conversely, can we say something about the active site of a group of CDK's (potentially only 1) based on which type of ligands that binds well. i.e. all the ligands binding well have a carboxylic acid group or a benzene ring in common.
* (BONUS) Can we make a regression model (QSAR) to predict the binding affinity between CDK proteins and ligands.  

Could we predict from any random protein its potential affinity with CDK? (we could use QSAR or simply prediction methods???)



### Methods


### Proposed timeline
#### *How to achieve this (in steps):*

- Preprocess: 
    - only looking at relevant (47) CDK and their binding ligands (suppose it will be enough data? we can add more data for milestone 3)
    - Find all unique binding ligands (4500?)
    - Create four dataframes for each of the measures of binding affinities for CDKs and ligands (will each be of dimension 47x4500?).
    - Assign labels to the affinity scores for three levels of affinity (Ki_weak, Ki_moderate, Ki_strong, EC50_weak, EC_moderate , ... etc. 12 labels in total) Julie and Mathilde might find a good source to indicate this. We will have to define a good (and fair!) mapping and justify it!
- Construct dataframe: match unique pairs ligands-CDKs -> make dataframe with the affinity measures and labels as rows
- Maybe dimensionality reduction: PCA (but we dont have so many parameters so might not make sense)
- Clustering/grouping: group on similar ligands (from a bioperspective view - how to? database?)
- Look into the Regression model if time permits.





### Organization within the team

A list of internal milestones up until project Milestone P3.
Create a gantt chart. 
Link it **here**

And then it all makes sense, great job, amazing job, couldn't have done it better 6/6 the world and the universe is saved GG


### Questions for TAs (optional)

Add here any questions you have for us related to the proposed project.
