# Tracking enhanced approaches and their visualization

### Description
There are total of 9 experimental approaches to test the trackers performance on dual modality streams. Each of those approaches is represented visually in `Final visualization of results.ipynb` and briefly described. Those approaches are further implemented in a single file `Methods implementation.ipynb`. If you have the Camel dataset with the same structure as mentioned [here](#project-structure) all of the methods will be run on all of the files.

# Dependencies
Extract `Pretrained-models.zip` and move its content to the working directory. Following files should be present:
- `model_a.pkl`
- `scaler_a.pkl`
- `imputer_a.pkl`

- `model_b.pkl`
- `scaler_b.pkl`
- `imputer_b.pkl`

- `model_best1.pkl`
- `model_best2.pkl`
- `scaler_best.pkl`
- `imputer_best.pkl`

Please not that files with suffix '_a' are the same as the ones without suffix, but this was made to keep all the models together if this file was to be split later. 

# Camel Dataset Feature Extraction and Model Training

## Description
This project consists of three Jupyter notebooks designed for feature extraction from the Camel dataset, model training, and implementing a decision tree. The Camel dataset is required to be in a specific directory structure for the notebooks to function correctly.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Feature Extraction](#feature-extraction)
  - [Model Training](#model-training)
  - [Decision Tree Implementation](#decision-tree-implementation)
- [Dependencies](#dependencies)

## Project Structure
The project includes the following Jupyter notebooks:
1. `Feature extraction for decision tree.ipynb`: For feature extraction from the Camel dataset.
2. `Machine learning for modality selection.ipynb`: For training the model on the Camel dataset.
3. `Decision tree implementation.ipynb`: For implementing the decision tree.

The Camel dataset should be placed in a directory named `Camel` with the following structure:

Camel/
├── seq-1/
│   ├── Seq1-Abs
│   ├── Visual-seq1
│   └── IR-seq1
├── seq-2/
│   ├── Seq2-Abs
│   ├── Visual-seq2
│   └── IR-seq2
├── seq-3/
│   ├── Seq3-Abs
│   ├── Visual-seq3
│   └── IR-seq3 



## Installation
1. Clone the repository:
   ```bash
   git clone https://gitlab.workswell.cz/tracking/tracking-implementation.git
   cd tracking-implementation
2. Ensure you have Python installed. It's recommended to use a virtual environment.
3. Install the required libraries:
    pip install -r requirements.txt
    
## Usage

### Pretrained Models
If you want to skip the training and feature extraction steps, you can use the following pre-trained files:
- `model.pkl`
- `scaler.pkl`
- `imputer.pkl`

These files contain the pretrained model, scaler, and imputer, respectively. Ensure these files are in the appropriate directory before running the decision tree implementation notebook.

### Feature Extraction
1. Open `Feature extraction for decision tree.ipynb` in Jupyter Notebook.
2. Run the cells to extract features from the Camel dataset.

### Model Training
1. Open `Machine learning for modality selection.ipynb` in Jupyter Notebook.
2. Run the cells to train the model using the extracted features.

### Decision Tree Implementation
1. Open `Decision tree implementation.ipynb` in Jupyter Notebook.
2. Run the cells to implement and evaluate the decision tree model.


## Dependencies
The following libraries are used in the project:
```python
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import warnings

# Suppress UserWarning related to feature names
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")
