Table of Contents

Key Scripts

Reproduce.m: Replicates results presented in the paper (Figures 3 and 4) and the appendix (Figure 20).
DownloadDatasets.m: Retrieves datasets from GitHub, OpenML, and UCI, then processes them.
fitPTC.m: Implements the training algorithm for the Prototype Theory Classifier (Paper's Algorithm 1).
NaivePrototype.m: A brute-force training algorithm for Prototype Theory Classification (inefficient).
demo_iris_brute_force.m: Demonstrates the brute-force approach on the IRIS dataset.

Additional Scripts

fitNCC.m: Implements the Nearest-Centroid Classifier (NCC).
fitSNCC.m: Implements the Sparse Nearest-Centroid Classifier (SNCC).
predNCC.m: Performs predictions using the NCC model.
predSNCC.m: Performs predictions using the SNCC model.

Utilities

eval_perf.m: Provides evaluation metrics such as recall, balanced accuracy, and F-measure.
datasets.mat: Contains metadata for the datasets.
datasets.xlsx: Detailed dataset information, including URLs and references to related papers.
clearex.m: An external script by Arnaud Laurent to clear all variables in memory except specified ones.
arff2table.m: Converts .arff files into MATLAB table format.