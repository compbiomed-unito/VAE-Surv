# VAE-Surv: Variational Autoencoder for Survival Analysis

## Introduction

**VAE-Surv** integrates a **Variational Autoencoder (VAE)** with a **survival prediction model** to identify patient subgroups and predict survival outcomes. Originally developed for **Myelodysplastic Syndromes (MDS)** patients, it can be applied to various biomedical datasets. The VAE reduces high-dimensional molecular data to a **latent space**, which is then used alongside clinical features in a **Cox neural network** for survival analysis.

### Key Features:
- **Dimensionality Reduction**: Extracts a meaningful latent representation from high-dimensional genomic data.
- **Survival Prediction**: Uses a deep Cox network to model survival risk.
- **Patient Clustering**: Performs unsupervised clustering in the latent space to identify distinct prognostic groups.
- **Joint Training**: Optimizes both reconstruction (VAE) and survival loss to enhance predictive performance.

The repository includes an example on the **METABRIC breast cancer dataset**, demonstrating data preprocessing, training, and evaluation.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/VAE-Surv.git  
   cd VAE-Surv
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Run Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

## Usage




Rollo, Cesare, et al. "VAE-Surv: A novel approach for genetic-based clustering and prognosis prediction in myelodysplastic syndromes." Computer Methods and Programs in Biomedicine (2025): 108605.
