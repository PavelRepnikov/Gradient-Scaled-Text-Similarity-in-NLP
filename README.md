# Text Similarity Prediction

This project aims to predict the similarity score between pairs of texts, where the score ranges from -5 to 5. The dataset contains queries and positive/negative responses, and the task is to train a model that accurately predicts the similarity score between two pieces of text.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)

## Project Structure
- `train.py`: Main script for training the model.
- `model.py`: Model architecture and utility functions.
- `utils.py`: Helper functions for data preprocessing and model evaluation.
- `README.md`: Project documentation.

## Dataset
The dataset consists of queries and several columns for positive and negative responses. The data is reformatted into pairs of texts with corresponding similarity scores for training.

### Original Dataset Format:
| query | pos | neg  |
|-------|-------|-------|
| text  | text  | text  |

### Dataset Format After Initial Prepocessing By Preprocess.ipynb:
| query | pos_0 | pos_1 | pos_2 | pos_3 | pos_4 | neg_0 | neg_1 | neg_2 | neg_3 | neg_4 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| text  | text  | text  | text  | text  | text  | text  | text  | text  | text  | text  |

### Reformatted Dataset Format:
| text1  | text2  | score |
|--------|--------|-------|
| query  | pos_0  | 5     |
| query  | neg_0  | -5    |

## Model
The model is a neural network based on pre-trained transformers to compute the similarity between two input texts. It outputs a scalar score that represents the predicted similarity.

- Pre-trained transformer model: `BERT`
- Optimizer: `AdamW`
- Loss function: `Mean Squared Error (MSE)`

### GPU Setup
The model supports multi-GPU training. Learned on 2 3060 ti:
- `per_device_train_batch_size=32` on `cuda:0` and `cuda:1`

## Training
To train the model, run the `train.ipynb` script. The model will process the data and calculate the loss on both training and validation sets.

## Training Metrics:

Training Loss: Measures how well the model fits the training data.
Validation Loss: Measures the model's performance on unseen data.
RMSE (Root Mean Squared Error): Average magnitude of prediction errors.
MAE (Mean Absolute Error): Average absolute difference between predicted and actual values.

## Usage

Once trained, the model can be used to predict the similarity between any two texts. The model will output a score between 0 and 5, indicating how similar the texts are.

## Minuses

As the code was written by an illiterate monkey, it has some minuses.

Firstly, i started writing the project with hopes of achieving a metric with -5;5 borders, and achieved 0;5.

Secondly, i used regression instead of classification to give to initial metrics.

Then, the data has equal amount of negative and positive examples. In real-world cases, the ratio of positive to negative examples might be imbalanced, leading to decreased model performance.

Then, even on 2 GPUs, i might have had insufficient resourses to efficiently train the model.

Then, thera are a lot of things that i may have not consdered - i am but a beginner in a world of ML.

Overall, i dont recommend code and model is used for serious work without some heavy touch up. If you stumbled upon this code and even read the readme to the end, use it mainly as a way to laugh at me.

The main problem of a project was insufficient time given to complete the project; alas this is not a serious work.