# Causal Graph Discovery from Genomic Data in Health and Alzheimer’s Disease

This repository contains the source code and resources for the master dissertation exploring causal relationships in genomic data in Alzheimer’s Disease.

The project is structured into a two-stage pipeline:
1.  **Feature Selection:** High-dimensional genomic data often contains thousands of features (genes) but a limited number of samples. This stage employs feature selection techniques to reduce dimensionality, remove noise, and select a subset of salient genes relevant to the disease state.
2.  **Causal Discovery:** Using the selected features, this stage applies causal discovery algorithms to construct a Causal Directed Acyclic Graph (DAG). This graph represents the causal dependencies between genes, providing insights into potential regulatory pathways.

The genomic dataset used for this study is available for download from Google Drive. Please place the downloaded file(s) into a `data/` directory at the root of the project.
- **[Download Dataset](https://drive.google.com/file/d/1gfiDuyuBXJOkWckkShbn-YjNdHIhNbKl/view?usp=sharing)**
