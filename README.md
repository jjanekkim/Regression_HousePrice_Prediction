# House Price Regression Project
## Overview
This repository comprises Jupyter Notebooks, a Python package, and a dedicated pipeline designed for accurate house price prediction.

## Table of Contents
* [Introduction](#introduction)
* [General Info](#general-info)
* [Dependencies](#dependencies)
* [Project Structure](#project-structure)
* [Utilization](#utilization)

## Introduction
Welcome to the United States House Price Prediction project! The primary aim of this project is to create a robust and generalized model for predicting house prices in the United States. This project addresses the prevalent issue of data leakage often encountered in Kaggle competitions, ensuring a meticulous and accurate approach to modeling.

The project is structured into distinct sections, including exploratory data analysis (EDA) and model training notebooks, validation and testing notebooks, a specialized Python package dedicated to house price prediction, and an efficient pipeline for streamlined processes.

Through the development of the house price prediction model, our goal is to offer valuable insights into the dynamics of real estate pricing trends over time. Additionally, this model seeks to assist potential homebuyers in identifying the opportune timing for making house purchases, providing informed and data-driven guidance in the housing market.

## General Info

This project tackles the critical issue of data leakage commonly encountered in competitions like Kaggle. Data leakage involves the unintended or improper inclusion of information from outside the training data during model development. This could lead to overfitting, resulting in an inflated performance on the model's accuracy and hindering its effectiveness when applied to real-world 'test' data.

To mitigate the risk of data leakage, I rigorously segregated the train, validation, and test datasets. This strict separation ensures that unseen or future data remains inaccessible during model training, promoting a more reliable and robust model.

The development of a generalized model notably improved the RMSE from 24K to 18K, underscoring the efficacy of a generalized approach. Notably, achieving the highest score in a Kaggle competition might not necessitate splitting and generating additional validation data. Instead, addressing data leakage and constructing a generalized model can significantly enhance the model's performance when dealing with real-world data. This practice also fosters a habit of thorough data comprehension and effective model building.

## Dependencies
This project is created with:
- Python version: 3.11.3
- Pandas package version: 2.0.1
- Numpy package version: 1.24.3
- Matplotlib package version: 3.7.1
- Seaborn package version: 0.12.2
- Scikit-learn package version: 1.2.2
- XGBoost version: 1.7.6

## Project Structure
- **Data**: Access the dataset [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data]).
- **DataAnalysis_ModelTrain**: Jupyter notebook encapsulates crucial steps for the project, encompassing data analysis, preparation, and model training. It encompasses processes for splitting the dataset into training and validation sets, comprehensive visualization techniques, data scaling, and meticulous cleaning procedures.
- **Validation_Test**: Jupyter notebook comprises sections for predicting using the validation and test dataset.
- **houseprice_package**: Python file containing essential functions utilized throughout the project.
- **hp_train_pipeline**: Pipeline for the model training, outlining the workflow and stages involved in the project.

## Utilization
To utilize this project, please download the dataset from the provided link mentioned above. Subsequently, download the 'houseprice_package.py' file and execute it on Jupyter notebook.
