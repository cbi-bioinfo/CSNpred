# Cell-specific network-based cell type prediction via graph convolutional network using transcriptomics profiles
The present work proposes CSNpred, a cell type prediction framework that leverages cell-specific networks and graph convolutional networks (GCNs), which can be utilized for both scRNA-seq and SRT data. Our model constructs a cell-specific network for each cell by identifying neighbors with similar gene expression patterns and spatial proximity (when available) to form an undirected weighted graph. The GCN-based classification module then extracts node embeddings, which are subsequently used for cell type prediction. 

## Requirements
* Pytorch (v1.6.0)
* Python (>= 3.6)
* Python packages : numpy, pandas, os, sys
