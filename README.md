# README for Mathematical Finance Project

## Overview
This project focuses on the pricing and hedging of American-style options using deep learning techniques. The primary goal is to develop a method that learns optimal exercise behavior, prices, and hedging strategies from samples of underlying risk factors.

## Authors
- **Taha Akbari**
- **Mahdi Hajialilue**
- **Spring 2023**
- **Math Department, Sharif University of Technology**

## Goals
- Develop a deep learning method to learn optimal exercise behavior and pricing strategies.
- Calculate continuation values and design a stopping rule based on comparing current payoffs with continuation values.
- Compute lower and upper bounds for option prices and derive hedging strategies.

## Project Structure
- **ML_Template__3__7.pdf**: This document contains theoretical foundations, mathematical formulations, and methodologies for pricing and hedging American-style options.
- **mathematical_finance_project.ipynb**: This Jupyter Notebook implements the concepts discussed in the PDF, providing a practical application of the theoretical models.

## Key Concepts from ML_Template__3__7.pdf

### Mathematical Formulations
1. **Hedging Until the First Possible Exercise Time**:
   - The document discusses the computation of hedging strategies until a specified time \( t_1 \). If the option remains valid at \( t_1 \), the strategy can be extended to \( t_2 \).
   - The value of the option at time \( t_1 \) is approximated using the function \( V_{t_1}^\Theta = v^{\theta_1}(X_1) \).

2. **Mean Squared Error Minimization**:
   - The goal is to find hedging positions \( h_m \) that minimize the mean squared error defined as:
     $$
     E[(\widehat{V} + (h.P)_{t_1} - V_{t_1}^\Theta)]
     $$

3. **Neural Network Approximation**:
   - Instead of searching the entire space for functions \( h_m \), the document proposes using neural networks \( h^\lambda: \mathbb{R}^d \to \mathbb{R}^e \) to approximate these functions.
   - The optimization problem is defined as:
     $$
     \sum_{k=1}^{K_H} (\hat{V} + \sum_{m=0}^{M-1} h^{\lambda_m}(y^k_m) \cdot (p_{m+1}(y^k_{m+1}) - p_m(y^k_m)) - v^{\theta_1}(y_M^k))^2
     $$
   - The training of the networks is performed using the Adam optimization method.

### Content Summary
- The document provides a comprehensive overview of the theoretical aspects of pricing and hedging options, focusing on the use of deep learning for approximating complex functions.
- It includes mathematical expressions and methodologies that are essential for understanding the underlying principles of the project.

## Jupyter Notebook: mathematical_finance_project (1).ipynb
This notebook implements the theoretical concepts outlined in the PDF. It includes the following sections:

1. **Data Preparation**: Loading and preprocessing the data necessary for training the neural networks.
2. **Model Definition**: Defining the architecture of the neural networks used for approximating the hedging functions.
3. **Training Process**: Implementing the training loop using the Adam optimizer to minimize the defined loss function.
4. **Evaluation**: Assessing the performance of the trained models and visualizing the results.

### Data Structure
The dataset used in the notebook likely contains the following columns (this is a general assumption based on typical financial datasets):
- **Date**: The date of the financial record.
- **Open**: The opening price of the asset.
- **High**: The highest price during the trading period.
- **Low**: The lowest price during the trading period.
- **Close**: The closing price of the asset.
- **Volume**: The number of shares traded.

## Requirements
To run this project, ensure you have the following Python packages installed:

- numpy
- pandas
- matplotlib
- seaborn
- tensorflow (or keras)

You can install these packages using pip:

```bash
pip install numpy pandas matplotlib seaborn tensorflow
