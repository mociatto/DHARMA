# DHARMA: Distributed Hybrid Architecture for Responsible Model Alignment and Assured Privacy

[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/dharma-01-foundation)

A comprehensive security audit framework for Hybrid Vertical Federated Learning (VFL) systems in medical imaging, evaluating both integrity and privacy vulnerabilities through adversarial attack analysis.

## Overview

DHARMA implements a **Hybrid Vertical Federated Learning** system that mirrors real-world clinical workflows for skin lesion classification. The framework consists of two privacy-preserving clients that collaborate through a central server without sharing raw patient data:

- **Image Client**: Represents edge hardware (e.g., Digital Dermatoscope) that captures dermoscopic images directly from patients
- **Tabular Client**: Represents the hospital's Electronic Health Record (EHR) system, holding secure metadata (Age, Sex, Anatomical Site)

In VFL, clients exchange feature embeddings that are mathematically rich representations containing highly condensed information, rather than raw data. The central server aggregates these embeddings using cross-modal attention mechanisms to make final predictions.

![DHARMA Architecture](images/Fig-1.png)

## Dataset

The framework is evaluated on the **HAM10000** dataset, which contains dermoscopic skin lesion images with extreme class imbalance. Common conditions like Melanocytic nevi (`nv`) vastly outnumber rare but critical cases like Melanoma (`mel`), presenting a significant challenge for medical AI systems.

## Architecture

The DHARMA architecture implements **disentangled representation learning** where each client produces two types of embeddings:

- **Specific embeddings**: Capture modality-specific information (image-specific or tabular-specific features)
- **Invariant embeddings**: Capture cross-modal shared information that enables alignment between image and tabular representations

The transformer server aggregates these embeddings using multi-head attention mechanisms to enable cross-modal fusion and final classification.

## Notebooks

The project is organized into three sequential notebooks:

### 1. Foundation Notebook

[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/dharma-01-foundation)

Establishes the baseline Hybrid VFL model by:
- Training dual image and tabular clients with disentangled representations
- Training a transformer-based server for cross-modal aggregation
- Computing and caching embeddings for both training and test sets
- Evaluating baseline performance metrics (accuracy, balanced accuracy, F1-score, precision, recall)

**Outputs**: Pre-trained models, computed embeddings, and baseline performance metrics used by subsequent audit notebooks.

### 2. Integrity Audit Notebook

[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/dharma-02-integrity-audit)

Evaluates the system's robustness against integrity attacks:

- **Untargeted Noise Attack**: Simulates sensor noise by adding Gaussian noise to image embeddings
- **Targeted Label-Flipping Attack**: Poisoning attack that flips labels from source class (Melanoma) to target class (Nevus)
- **Targeted Backdoor Attack**: Injects visual triggers into images to cause misclassification to a target class
- **WaveAttack**: Frequency-domain attack using wavelet transforms

**Metrics**: Attack success rate, accuracy degradation, per-class performance analysis.

### 3. Privacy Audit Notebook

[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/dharma-03-privacy-audit)

Analyzes privacy leakage through inference and reconstruction attacks:

- **Label Inference Attack**: Infers private labels from captured gradients during federated learning
- **Membership Inference Attack**: Determines whether a patient sample was part of the training set
- **Data Reconstruction Attack**: 
  - **Image Reconstruction**: Reconstructs images from image-specific embeddings (model inversion)
  - **Attribute Inference**: Infers demographic attributes (age, sex) from tabular embeddings

**Metrics**: Attack accuracy, AUC, TPR@low FPR, reconstruction quality (MSE, MAE, correlation).

## Key Features

- **Modular Design**: Three separate notebooks for baseline establishment, integrity analysis, and privacy analysis
- **Comprehensive Security Evaluation**: Covers both integrity (model robustness) and privacy (data leakage) vulnerabilities
- **Real-World Scenarios**: Attacks model realistic threat scenarios in medical federated learning
- **Reproducible**: Caching mechanisms enable efficient experimentation and reproducibility
- **Medical Context**: Specifically designed for skin lesion classification with class imbalance considerations

## Technical Stack

- **Deep Learning**: TensorFlow/Keras for model architecture and training
- **Federated Learning**: Custom VFL implementation with disentangled representations
- **Security Analysis**: Adversarial attack implementations and evaluation metrics
- **Visualization**: Matplotlib, Seaborn for results visualization
- **Data Processing**: NumPy, Pandas for data handling and analysis

## Results

The framework demonstrates:

- Baseline model performance on imbalanced medical dataset
- Vulnerability assessment across multiple attack vectors
- Privacy leakage quantification through various inference attacks
- Reconstruction quality analysis for both image and tabular data

## Citation

If you use this framework in your research, please cite:

```
DHARMA: Distributed Hybrid Architecture for Responsible Model Alignment and Assured Privacy
```

## License

This project is provided for research and educational purposes.

