# Transfer Learning with AlexNet and VGG16 on CIFAR-10  
### Fine-Tuning vs. Feature Extraction Comparative Study

---

## Overview

This project presents a comparative study on transfer learning performance using two well-known convolutional neural networks — AlexNet and VGG16 — applied to the CIFAR-10 image classification dataset.  
The study examines and contrasts fine-tuning and feature extraction strategies, analyzing how much pre-trained convolutional features contribute to classification accuracy when transferred to a smaller dataset.

The work demonstrates a systematic approach to:
- Re-using ImageNet-trained models  
- Modifying architectures for a 10-class dataset  
- Evaluating the trade-offs between training efficiency and model adaptability

---

## Objectives

- Implement and train AlexNet and VGG16 models using two transfer learning strategies:  
  - Feature Extraction: Freeze convolutional layers and train only the fully connected layers  
  - Fine-Tuning: Unfreeze deeper convolutional layers for retraining  
- Evaluate both strategies on the CIFAR-10 dataset  
- Compare training time, accuracy, and loss performance  
- Visualize learning curves and analyze key results  

---

## Background

Transfer learning leverages prior knowledge learned from a large dataset (e.g., ImageNet) to solve related tasks on smaller datasets (like CIFAR-10).  
Two major strategies are explored:

1. **Feature Extraction**  
   - The convolutional base is frozen to preserve pre-trained filters  
   - Only the final classifier is retrained for the new task  

2. **Fine-Tuning**  
   - Some (or all) convolutional layers are unfrozen  
   - The model refines deeper representations to adapt to CIFAR-10’s domain  

This approach saves time, reduces overfitting risk, and improves model generalization.

---

## Project Structure

Transfer-Learning-with-AlexNet-and-VGG16-on-CIFAR-10/
│
├── 1203022CaseStudy.ipynb      # Main notebook containing all implementation steps  
├── README.md                   # Project documentation (this file)  
├── requirements.txt (optional) # Dependencies list  
└── /models                     # Saved trained models (optional)

---

## Installation and Setup

### 1. Clone the Repository
git clone https://github.com/msalem02/Transfer-Learning-with-AlexNet-and-VGG16-on-CIFAR-10-Fine-Tuning-vs.-Feature-Extraction.git
cd Transfer-Learning-with-AlexNet-and-VGG16-on-CIFAR-10-Fine-Tuning-vs.-Feature-Extraction

### 2. Install Required Packages
If you have a requirements.txt, use:
pip install -r requirements.txt

Otherwise, manually install:
pip install torch torchvision matplotlib numpy

### 3. Run the Notebook
Open the Jupyter Notebook:
jupyter notebook 1203022CaseStudy.ipynb

Execute the notebook cells in order — from data loading to evaluation — to reproduce the full experiment.

---

## Dataset

CIFAR-10 Dataset  
- 60,000 color images (32×32 pixels)  
- 10 balanced classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
- Split: 50,000 training images + 10,000 test images  

All images are normalized and optionally augmented before training.

---

## Implementation Details

### Pretrained Architectures
- AlexNet and VGG16 imported from torchvision.models pretrained on ImageNet  
- Final classification layer replaced with a Linear(4096, 10) output head for CIFAR-10

### Training Strategies
1. Feature Extraction
   - Freeze convolutional layers (requires_grad=False)
   - Train only fully connected layers  

2. Fine-Tuning
   - Unfreeze deeper layers (typically last 2–3 conv blocks)
   - Train entire model with smaller learning rate  

### Training Configuration
- Optimizer: Adam  
- Loss Function: Cross-Entropy Loss  
- Batch Size: 64  
- Epochs: 20  
- Learning Rate: 0.001 (fine-tuning often uses smaller LR)

---

## Results Summary

| Model   | Strategy           | Test Accuracy | Training Time | Observation |
|----------|-------------------|----------------|----------------|--------------|
| AlexNet  | Feature Extraction | 81.32 %        | Fast           | Stable, less adaptable |
| AlexNet  | Fine-Tuning        | 89.17 %        | Moderate       | Significant improvement |
| VGG16    | Feature Extraction | 81.96 %        | Fast           | Slightly better than AlexNet |
| VGG16    | Fine-Tuning        | 92.23 %        | Higher         | Best overall performance |

Training and validation curves illustrate:
- Faster convergence for fine-tuned models  
- Lower final loss  
- Better generalization on unseen CIFAR-10 data  

(Plots and images are included inside the Jupyter Notebook.)

---

## Key Insights

- Fine-tuning improves accuracy by enabling adaptation of low-level and mid-level filters to the new dataset.  
- VGG16, being deeper than AlexNet, benefits more from fine-tuning at the cost of longer training time.  
- Feature extraction is efficient but limited when source and target datasets differ significantly.  
- Transfer learning drastically reduces training data and time requirements compared to training from scratch.

---

## Future Enhancements

- Implement learning rate scheduling and early stopping  
- Test newer architectures (ResNet, DenseNet, EfficientNet)  
- Explore data augmentation and regularization for generalization  
- Experiment with CIFAR-100 or custom datasets

---

## Example Results

(Insert confusion matrices, accuracy/loss plots, or sample predictions here for visual clarity.)

---

## Author

Mohammed Yousef Salem  

- Email: salemmohamad926@gmail.com   
- LinkedIn: https://www.linkedin.com/in/msalem02  
- GitHub: https://github.com/msalem02

---

## License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute it for educational or research purposes.  

See the LICENSE file for details.

---

## Acknowledgements

- PyTorch and Torchvision libraries  
- CIFAR-10 dataset by Alex Krizhevsky  
- Open-source community for pretrained architectures  
- Academic supervision and guidance from Birzeit University faculty  

---

## Contribute and Support

If you found this project useful or educational:
- Give it a star on GitHub  
- Share it with others interested in Deep Learning and Transfer Learning
