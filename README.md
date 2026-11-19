# Road Surface Condition Classification Using Deep Learning

Damaged road surfaces such as cracks and other surface defects can introduce safety concerns and require timely identification to support maintenance. Since manual road inspection can be time-consuming and labor-intensive, deep learning algorithms can be used to automatically classify road images.

This project performs binary road-surface classification into:

- Normal
- Damaged

The project uses the BDRoad-Sense dataset and compares two primary models:

1. Custom CNN baseline
2. MobileNetV2 with transfer learning

In addition, two published methodologies were adapted and evaluated using the same BDRoad-Sense dataset:

- ResNet50 based on Li et al. [1]
- RoadNet based on Alqethami et al. [4]

## Dataset

The original BDRoad-Sense dataset contains five classes:

- Normal Road
- Minor Damage
- Major Damage
- Manhole
- Speed Breaker

For this project:

- Normal Road → Normal
- Minor Damage + Major Damage → Damaged
- Manhole and Speed Breaker were excluded

The original BDRoad-Sense dataset can be accessed at:

https://data.mendeley.com/datasets/z3nx8n396g/3

## Models

### Custom CNN

A four-layer convolutional neural network is used as the baseline model.

### MobileNetV2

A pretrained MobileNetV2 model is used with ImageNet weights and transfer learning for binary classification.

### Adapted ResNet50

The ResNet50 methodology from Li et al. was adapted and evaluated using the BDRoad-Sense dataset.

### Adapted RoadNet

The RoadNet methodology from Alqethami et al. was adapted and evaluated using the BDRoad-Sense dataset. Class-weighted Cross Entropy Loss was used to reduce the effect of class imbalance.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The Damaged class is treated as the positive class.

## Robustness Evaluation

The primary models are also evaluated under:

- Clean images
- Low brightness
- High brightness
- Mild blur
- Strong blur

## Reproducibility

A fixed random seed of 42 is used during the experiments.

The repository includes:

- Model implementation notebooks
- Dataset split information
- Architecture diagram
- Dependency list
- Code required to reproduce the reported experiments

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```
