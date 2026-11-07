# Arabic Handwritten Character Recognition (AHCD)

This is our project proposal for the deep learning course. The goal is to classify images of handwritten Arabic letters into the correct letter class.

Team: Abdulaziz Alguraini, Bader Alqahatani

## The problem

We're using the Arabic Handwritten Characters Dataset (AHCD). Given a 32x32 grayscale image of a single handwritten Arabic letter, the model should predict which of the 28 letters it is. This is a multi-class image classification problem.

We picked this because Arabic handwriting recognition is less developed than English/Latin handwriting recognition, and it could be useful for things like digitizing handwritten Arabic documents and forms.

## Dataset

AHCD has 16,800 images total, written by 60 people. It comes with an official split of 13,440 training images and 3,360 test images, 480 and 120 per letter respectively, so the classes are perfectly balanced. Each image is 32x32 pixels, grayscale, and the data is provided as CSV files where each row is a flattened image.

- Download it here: https://www.kaggle.com/mloey1/ahcd1 (you need a free Kaggle account)
- License: [check the Kaggle page sidebar and put it here]
- Source: El-Sawy, A., Loey, M., & El-Bakry, H. (2017). Arabic Handwritten Characters Recognition using Convolutional Neural Network. WSEAS Transactions on Computer Research, 5.

The dataset itself isn't included in this repo, you have to download it separately (see below).

## How to run this

1. Download and unzip the dataset from the Kaggle link above.
2. Put the CSV files in a `data/` folder in the project root.
3. Install the requirements:
   ```
   python -m pip install -r requirements.txt
   ```
4. Run the notebook:
   ```
   python -m notebook ahcd_exploration.ipynb
   ```
   and run all cells.

Milestone 3 needs a GPU (it trains VGG19, ResNet18 and MobileNetV2). We ran `milestone3_cnn.ipynb` on Google Colab: upload the notebook, switch the runtime to a T4 GPU, create a `data` folder with the four CSV files, and use Runtime → Run all. It takes about 30–45 minutes.

## What's in this repo

- `ahcd_exploration.ipynb` — loads the data, checks it, makes the plots, and creates the train/val split
- `milestone2_mlp.ipynb` — Milestone 2: PyTorch data pipeline, shallow (2 hidden layers) and deep (5 hidden layers) MLPs, loss/optimizer/learning rate comparisons
- `milestone3_cnn.ipynb` — Milestone 3: CNN from scratch (augmentation, tuning, early stopping) and transfer learning with VGG19, ResNet18 and MobileNetV2, plus SVM / random forest / logistic regression on the pretrained features. We ran it on Google Colab with a GPU
- `logs/` — per-epoch training logs (CSV) from Milestone 3
- `checkpoints/` — the selected CNN checkpoint (`cnn_best.pt`) and the configuration files. The pretrained checkpoints are too big for GitHub (VGG19 is over 500 MB), so they are not included
- `data/` — where the dataset CSVs go (not included, you download it yourself)
- `figures/` — the plots the notebook generates
- `splits/` — the train/validation split indices, so it's reproducible
- `results/` — a summary of the dataset stats

## Split and metric

We're keeping the official test set (3,360 images) untouched until final evaluation. From the official training set, we split off 15% for validation (stratified by class, random seed 42), leaving 85% for actual training. That's 11,424 train / 2,016 validation / 3,360 test.

From Milestone 2 on we use an 80/20 train/validation split instead, since the milestone asks for it. That's 10,752 train / 2,688 validation / 3,360 test.

For the evaluation metric we're using accuracy, since the classes are perfectly balanced.

## Results so far

All numbers are on the official AHCD test set (3,360 images).

| Milestone | Model | Test accuracy |
|---|---|---|
| 2 | Shallow MLP (2 hidden layers) | 0.7914 |
| 2 | Deep MLP (5 hidden layers) | 0.7345 |
| 3 | CNN from scratch (6 conv layers, early stopping) | **0.9845** |
| 3 | ResNet18, last 3 layers fine-tuned | 0.9357 |
| 3 | SVM on ResNet18 features | 0.9482 |

The CNN trained from scratch is our best model so far.
