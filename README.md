# Protein Secondary Structure Prediction

## Overview

This project focuses on predicting protein secondary structures such as alpha-helices, beta-sheets, and coils using a hybrid deep learning model. The proposed model combines a Convolutional Neural Network (CNN) and a Bidirectional Long Short-Term Memory (BiLSTM) network to improve prediction accuracy by leveraging both local sequence patterns and long-range dependencies.

## Project Structure

```
|-- data/                 # Dataset files
|-- models/               # Model definitions
|-- notebooks/            # Jupyter notebooks for experiments
|-- src/                  # Core implementation
|-- results/              # Saved models and evaluation results
|-- README.md             # Project description
|-- requirements.txt      # Dependencies
|-- train.py              # Training script
|-- evaluate.py           # Evaluation script
```

## Dataset

- **Source:** CullPDB dataset
- **Format:** Protein sequences and their features in NumPy arrays
- **Input Features:** Position-Specific Scoring Matrix (PSSM) encoding with 21 features per residue, including evolutionary and structural information.
- **Labels:** Secondary structure labels based on the DSSP classification (8 classes: L, B, E, G, I, H, S, T)

## Model Architecture

1. **CNN Layer:** Extracts local patterns from protein sequences, such as motifs and short-range structural interactions.
2. **BiLSTM Layer:** Captures long-range dependencies by processing sequences in both forward and backward directions.
3. **Fully Connected Layer:** Maps learned features to secondary structure labels using a softmax activation function.

## Training

- **Optimizer:** Adam (learning rate: 0.001, weight decay: 1e-4)
- **Loss Function:** Cross-entropy loss for multi-class classification
- **Training Epochs:** 25
- **Batch Size:** 32

### Data Handling

- **Train-Test-Validation Split:** 80/10/10
- **Masking:** Excludes padded residues during loss and accuracy calculations by applying binary masking to ensure valid predictions only.

## Evaluation Metrics

- **Training Loss:** Average loss over training batches.
- **Validation Loss:** Average loss on validation set.
- **Training Accuracy:** Percentage of correctly predicted residues in the training set.
- **Validation Accuracy:** Percentage of correctly predicted residues in the validation set.

## Results

- **Best Model Performance:**
  - **Test Accuracy:** 70.65%
  - **Best Configuration:** CNN + BiLSTM with Adam optimizer and weight decay.
  - **Comparative Models:**
    - BiLSTM only: 67.43% accuracy
    - CNN only: lower accuracy compared to the hybrid model

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Future Improvements

- **Feature Representation:** Explore advanced encodings such as embeddings or self-attention-based feature extractors.
- **Model Architecture:** Implement Transformer-based architectures for enhanced sequence modeling.
- **Data Augmentation:** Use larger and more diverse datasets to improve generalization.
- **Hyperparameter Tuning:** Conduct extensive tuning for optimal performance.

## References

- [Protein Secondary Structure - Wikipedia](https://en.wikipedia.org/wiki/Protein_secondary_structure)
- [Deep Learning for Protein Structure - Scientific Reports](https://doi.org/10.1038/srep18962)

**Contributors:** Shravani Shilimkar and Aditi Tarate

