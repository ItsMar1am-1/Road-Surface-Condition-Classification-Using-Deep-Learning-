# Road-Surface-Condition-Classification-Using-Deep-Learning-
Damaged roads, introduces major issues such as cracks and surface deterioration, require identification to facilitate maintenance. Since manual inspection is time consuming and labor intensive, deep learning algorithms can be employed to automatically classify road images. 

# Road Surface Condition Classification Using Deep Learning

This project performs binary road-surface classification into:
- Normal
- Damaged

The project uses the BDRoad-Sense dataset and compares:
1. Custom CNN baseline
2. MobileNetV2 with transfer learning

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

Only the original images were used.

## Models
### Custom CNN
A four-layer convolutional neural network is used as the baseline model.

### MobileNetV2
A pretrained MobileNetV2 model is used with transfer learning for binary classification.

## Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Robustness Evaluation
The models are also evaluated under:
- Clean images
- Low brightness
- High brightness
- Mild blur
- Strong blur

## Requirements
Python, PyTorch, torchvision, NumPy, scikit-learn, matplotlib, Pillow.

## Reproducibility
A fixed random seed of 42 is used during the experiments.
