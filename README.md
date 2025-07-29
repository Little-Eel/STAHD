# STAHD: a scalable and accurate method to detect spatial domains in high-resolution spatial transcriptomics data

![图片1](https://github.com/user-attachments/assets/6fc87200-a4d5-44ce-b6db-b66687adec7f)

- Motivation: Spatial transcriptomics (ST) enables the study of spatial heterogeneity in tissues. However, current methods struggle with large-scale, high-resolution data, leading to reduced efficiency and accuracy in detecting spatial domains. A scalable, precise solution is urgently needed.
- Results: We present STAHD, a scalable and efficient framework for spatial domain detection in ST data. Combining a graph attention autoencoder with multilevel k-way graph partitioning, STAHD decomposes large graphs into compact subgraphs and generates low-dimensional embeddings. This improves computational efficiency and clustering accuracy. Benchmarks on human and mouse datasets show STAHD outperforms existing methods and accurately reveals spatially distinct tumor microenvironments and functional regions.

# Installation

STAHD is built with Scanpy, PyTorch, and PyG, and supports both GPU (preferred) and CPU execution.
First clone the repository.

```
git clone https://github.com/Little-Eel/STAHD.git
cd STAligner-main
```

It's recommended to create a separate conda environment for running STAHD:

```
#create an environment called env_STAHD
conda create -n env_STAHD python=3.8

#activate your environment
conda activate env_STAHD
```

Install all the required packages.
For Linux

```
pip install -r requirement.txt
```

For MacOS

```
pip install -r requirement_for_macOS.txt
```

The use of the mclust algorithm requires the rpy2 package (Python) and the mclust package (R). See https://pypi.org/project/rpy2/ and https://cran.r-project.org/web/packages/mclust/index.html for detail.

The torch-geometric library is also required, please see the installation steps in https://github.com/pyg-team/pytorch_geometric#installation

Install STAHD.

```
python setup.py build
python setup.py install
```

# Datasets

All spatial transcriptomics datasets used in this study are publicly available. Detailed sources and download links are listed below:10x Visium human dorsolateral prefrontal cortex (DLPFC) da-taset and tutorials:
https://support.10xgenomics.com/spatial-gene-expression/datasets/1.2.0/V1_Human_DLPFC .

Xenium platform whole adult mouse brain dataset (xenium_whole_adult_mouse),including data and tutorials:
https://www.10xgenomics.com/datasets/xenium-prime-ffpe-neonatal-mouse. 

CosMx SMI human lymph node dataset (Cosmx lymph) from NanoS-tring:https://nanostring.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/cosmx-human-lymph-node-ffpe-dataset/. 

10x Genomics Visium-HD human breast cancer dataset (FFPE-IF):
https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-breast-cancer-ffpe-if. 

10x Genomics Visium-HD human tonsil dataset (fresh frozen, IF):
https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-tonsil-fresh-frozen-if
