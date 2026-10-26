# Bayesian Machine Learning: Classification 🐸 🐶

This repository contains the implementation of **Exercise 5: Classification** from the **Bayesian Machine Learning** course. It explores both Generative and Discriminative classification settings, complete with decision boundary visualization, Quadratic Discriminant Analysis (QDA), and Bayesian Linear Regression (BLR) applied to an image classification task (Dogs vs. Frogs).

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Tasks and Implementation Details](#tasks-and-implementation-details)
  - [1. Linear Decision Boundaries](#1-linear-decision-boundaries)
  - [2. Generative Image Classification (QDA)](#2-generative-image-classification-qda)
  - [3. Discriminative Image Classification (BLR)](#3-discriminative-image-classification-blr)
- [Requirements & Installation](#requirements--installation)
- [Usage](#usage)
- [Results](#results)

---

## 📖 Overview

The primary goal of this project is to implement and compare different Bayesian approaches to classification:
1.  **Generative Classification:** Fitting a Gaussian to data using a posterior over the mean, and establishing linear decision boundaries (LDA style).
2.  **Quadratic Discriminant Analysis (QDA):** Using an inverse-Wishart prior to estimate covariance matrices for classifying images.
3.  **Classification as Regression (BLR):** Utilizing Bayesian Linear Regression with Gaussian basis functions (RBFs) to perform discriminative classification and evaluate model uncertainty.

The image dataset used for Parts 2 and 3 consists of 6,000 dog images and 6,000 frog images extracted from the CIFAR-10 dataset.

The datasets are to large for Github. If you would like to access the data let me know and I will gladly share it.

---

## 📁 Project Structure

*   `ex5.py`: The main executable script containing the implementation for calculating posterior parameters, decision boundaries, QDA prediction, and BLR model fitting.
*   `ex5_utils.py`: A utility module containing the backbone classes and functions:
    *   `Gaussian`: Multivariate Gaussian model with an inverse-Wishart prior.
    *   `BayesianLinearRegression`: BLR model implementation using Minimum Mean Square Error (MMSE) estimates.
    *   `gaussian_basis_functions`: Generator for Gaussian basis (RBF) functions.
    *   Data loaders and plotting utilities.
*   `dogs.npy` & `frogs.npy`: (Required) Numpy arrays containing the image data for the classification tasks.

---

## 🧠 Tasks and Implementation Details

### 1. Linear Decision Boundaries
Visualizes the linear decision boundary between two classes generated from 2D Gaussian distributions.
*   Calculates the **MMSE estimate** for the class means under a Bayesian posterior.
*   Computes and plots the mean decision boundary based on the difference of the estimated means.
*   Samples 10 decision boundaries from the posterior distribution $\mu | \mathcal{D}$ to visualize the model's uncertainty regarding the exact location of the boundary.

### 2. Generative Image Classification (QDA)
Implements QDA to classify images of dogs and frogs.
*   Fits a multivariate Gaussian to each class using an **inverse-Wishart distribution** as a conjugate prior for the covariance matrix.
*   Evaluates the classifier across different values of $\nu$ (pseudo-observations of the scale matrix) to observe the transition from Maximum Likelihood (ML) solutions to Linear Discriminant Analysis (LDA) behaviors.
*   Classifies images by comparing the log-likelihoods: $\arg\max_c \log p(y=c | x)$.

### 3. Discriminative Image Classification (BLR)
Treats classification as a regression problem ($y \in \{-1, 1\}$).
*   Uses a **Bayesian Linear Regression** model with Gaussian basis functions centered around random subsets of the training data.
*   Evaluates accuracy over varying numbers of basis function samples ($M$).
*   Calculates the model's **uncertainty** for new test points using the ratio $d = |\mu_n / \sigma_n|$. Uses this metric to extract and plot the 25 most confident and 25 least confident predictions.

---

## ⚙️ Requirements & Installation

The project strictly utilizes fundamental scientific computing libraries. 

Install the required packages using `pip`:
```bash
pip install numpy matplotlib scipy
```

*Note: The project requires `matplotlib.use('TkAgg')` (configured in the utils file) for stable rendering in certain IDEs like PyCharm.*

---

## 🚀 Usage

1. Clone the repository to your local machine.
2. Ensure the dataset files (`dogs.npy` and `frogs.npy`) are placed in the root directory alongside the python scripts.
3. Run the main script:

```bash
python ex5.py
```

---

## 📊 Results

Running the script will sequentially output the following visualizations and console logs:
1.  **Decision Boundaries Plot:** A 2D scatter plot showing sampled points, the mean MMSE decision boundary (black), and 10 sampled boundaries from the posterior.
2.  **QDA Accuracy Plot:** A line graph demonstrating how train and test accuracy shift based on the prior parameter $\nu$.
3.  **BLR Accuracy Plot:** A line graph (log-scale X-axis) mapping classification accuracy against the number of Gaussian basis function centers ($M$).
4.  **Confidence Mosaics:** Two image mosaics rendering the 25 "least confident" and 25 "most confident" dog predictions, giving visual intuition into the model's uncertainty metric.
