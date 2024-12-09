# Cell-specific network-based cell type prediction via graph convolutional network using transcriptomics profiles
The present work proposes CSNpred, a cell type prediction framework that leverages cell-specific networks and graph convolutional networks (GCNs), which can be utilized for both scRNA-seq and SRT data. Our model constructs a cell-specific network for each cell by identifying neighbors with similar gene expression patterns and spatial proximity (when available) to form an undirected weighted graph. The GCN-based classification module then extracts node embeddings, which are subsequently used for cell type prediction. 

## Requirements
* Pytorch (v1.6.0)
* Python (>= 3.6)
* Python packages : numpy, pandas, os, sys, scikit-learn

## Usage
Clone the repository or download source code files.

## Inputs
### 1. Training Dataset
* train_X
  - Contains the gene expression profiles for the training dataset
  - Row : Sample, Column : Feature (Gene)
  - Shouldn't contain any sample_id column
  - If the user has spatial location information for each cell/spot, please make sure there are two columns with "array_row" and "array_col", representing the integer coordinates of row and column, respectively
  - Example :
```
A1BG,A1CF,...,A2ML1,array_row,array_col
0.342,0.044,...,0.112,1,2
...
```

* train_Y
  - Contains the integer-converted cell type information for the training dataset
  - The order of cell type should be same as **train_X**
  - The cell type label should always be in sequential format starting from 0
  - For example, if there are 5 cell types, each should be denoted as 0, 1, 2, 3, 4.
 
### 2. Testing Dataset
* test_X
  - Contains the gene expression profiles for the testing dataset
  - Row : Sample, Column : Feature (Gene)
  - Shouldn't contain any sample_id column
  - If the user has spatial location information for each cell/spot, please make sure there are two columns with "array_row" and "array_col", representing the integer coordinates of row and column, respectively
  - Format should be same as **train_X**
 

## How to run
1. Edit **"run_CSNpred.sh"** to make sure each variable indicate the corresponding train_X, train_Y and test_X dataset files as input.
2. Run the below command :
```
chmod +x run_CSNpred.sh
./run_CSNpred.sh
```

3. You will get an output **"prediction.csv"** with predicted cell type for test dataset.

## Contact
If you have any questions or problems, please contact to **joungmin AT vt.edu**.
