# Water-Quality-Prediction

🌊 Water Quality Prediction with Machine Learning

A sophisticated AI system that predicts water potability using advanced machine learning algorithms and real-time analysis.





🎯 Project Highlights

5 Advanced ML Models - From simple Logistic Regression to complex Random Forests
Interactive Visualizations - Beautiful insights with Plotly & Seaborn
Production-Ready Code - Optimized, documented, and ready to deploy
Zero Setup Required - Run instantly in Google Colab


The notebook is implemented in **Google Colab**, which provides an easy way to run Python code in the cloud without any installation required.

## Table of Contents

1. ## Description

This project uses machine learning to predict the potability of water based on various chemical and physical features. The dataset includes water quality parameters such as **pH**, **Sulfate**, and **Trihalomethanes**, and the goal is to classify water as either potable or non-potable. Using several machine learning algorithms, including Logistic Regression, Decision Trees, and Random Forest, we train and evaluate models to predict water potability.

2. [Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)



## How to Use

### 1. **Open the Project in Google Colab**:
Click the link below to open the project directly in **Google Colab**:

[Open in Google Colab](https://colab.research.google.com/github/your-username/your-repo/blob/main/your-notebook.ipynb)

### 2. **Run the Notebook**:
- Once opened in **Google Colab**, run each cell by clicking the **play button** on the left side of each code block.
- The notebook will guide you through the steps, from loading the dataset to training the machine learning models.

---

## Dependencies

All required libraries are pre-installed in **Google Colab**. However, if you are running the project locally, you will need to install the following libraries:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `seaborn`: For data visualization.
- `plotly`: For interactive visualizations.
- `matplotlib`: For static plotting.
- `scikit-learn`: For machine learning algorithms.
- `flask`: (optional) For deploying the model as a web application.




🛠️ Dependencies
### To install these dependencies locally, use the following command:



Copy# Core dependencies
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
seaborn>=0.11.0
plotly>=4.14.0
matplotlib>=3.3.0
```bash
pip install pandas numpy seaborn plotly matplotlib scikit-learn flask



Dataset
The dataset used in this project contains water quality data. The target variable is Potability, which indicates whether the water is potable (1) or not potable (0). The dataset consists of the following columns:

pH: pH level of the water.
Sulfate: Concentration of sulfate in the water.
Trihalomethanes: Concentration of trihalomethanes.
Potability: The target variable (1 = potable, 0 = not potable).
Dataset Sources:
The dataset is publicly available and can be loaded from your local machine or Google Drive.
If you're running this project in Google Colab, you can upload the dataset using:
python
Copy code


from google.colab import files
uploaded = files.upload()  # This will allow you to upload the dataset


Sure! Here’s a full README for your Water Potability Prediction machine learning project. I've included a description of the project, dependencies, instructions on how to run the notebook, and more. You can just copy-paste this into your README file.

Water Potability Prediction
Description
This machine learning project aims to predict whether water is potable (safe for drinking) or not based on various water quality features. The dataset contains information about the levels of different water quality parameters such as pH, Sulfate, Trihalomethanes, and other chemical compounds that impact water potability. By applying data preprocessing techniques, feature scaling, and exploring various machine learning models, this project demonstrates how different algorithms can be used to classify water as potable or non-potable.

The notebook is implemented in Google Colab, which makes it easy to run without any local setup. The project includes the following steps:

### Exploratory Data Analysis (EDA): Understanding the dataset, visualizing missing values, and exploring correlations between features.



Data Preprocessing: Handling missing values, scaling features, and preparing the dataset for training.
Model Training: Training multiple machine learning models, including:
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Model Evaluation: Evaluating model performance using accuracy scores and confusion matrices.
The project uses several popular Python libraries such as pandas, matplotlib, scikit-learn, and seaborn to handle the data and train the models.

Table of Contents
Description
How to Use
Dependencies
Dataset
Usage
Model Accuracy Comparison
License
How to Use
1. Open the Project in Google Colab:
Click the link below to open the project directly in Google Colab:

Open in Google Colab

2. Run the Notebook:
Once opened in Google Colab, run each cell by clicking the play button on the left side of each code block.
The notebook will guide you through the steps, from loading the dataset to training the machine learning models.
Dependencies
All required libraries are pre-installed in Google Colab. However, if you're running the project locally, you will need to install the following libraries:

pandas: For data manipulation and analysis.
numpy: For numerical computations.
seaborn: For data visualization.
plotly: For interactive visualizations.
matplotlib: For static plotting.
scikit-learn: For machine learning algorithms.
flask: (optional) For deploying the model as a web application.
To install these dependencies locally, use the following command:
bash
Copy code
pip install pandas numpy seaborn plotly matplotlib scikit-learn flask
Dataset
The dataset used in this project contains water quality data. The target variable is Potability, which indicates whether the water is potable (1) or not potable (0). The dataset consists of the following columns:

pH: pH level of the water.
Sulfate: Concentration of sulfate in the water.
Trihalomethanes: Concentration of trihalomethanes.
Potability: The target variable (1 = potable, 0 = not potable).
Dataset Sources:
The dataset is publicly available and can be loaded from your local machine or Google Drive.
If you're running this project in Google Colab, you can upload the dataset using:
python
Copy code
from google.colab import files
uploaded = files.upload()  # This will allow you to upload the dataset
Usage
1. Data Preprocessing & Exploratory Data Analysis (EDA)
In this step, we load the dataset, check for missing values, visualize correlations, and handle missing data by filling it with the mean of the column. The missing values in the features pH, Sulfate, and Trihalomethanes are filled with the column mean.
# Water Quality Prediction

## Description

This machine learning project predicts whether water is potable based on various features, such as pH level, Sulfate content, and Trihalomethanes concentration. The dataset is analyzed using exploratory data analysis (EDA), followed by preprocessing, model training, and evaluation using five different machine learning algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- KNN (K-Nearest Neighbors)
- SVM (Support Vector Machine)

The notebook is implemented in **Google Colab**, which provides an easy way to run Python code in the cloud without any installation required.

## Table of Contents

1. ## Description

This project uses machine learning to predict the potability of water based on various chemical and physical features. The dataset includes water quality parameters such as **pH**, **Sulfate**, and **Trihalomethanes**, and the goal is to classify water as either potable or non-potable. Using several machine learning algorithms, including Logistic Regression, Decision Trees, and Random Forest, we train and evaluate models to predict water potability.

2. [Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)



## How to Use

### 1. **Open the Project in Google Colab**:
Click the link below to open the project directly in **Google Colab**:

[Open in Google Colab](https://colab.research.google.com/github/your-username/your-repo/blob/main/your-notebook.ipynb)

### 2. **Run the Notebook**:
- Once opened in **Google Colab**, run each cell by clicking the **play button** on the left side of each code block.
- The notebook will guide you through the steps, from loading the dataset to training the machine learning models.

---

## Dependencies

All required libraries are pre-installed in **Google Colab**. However, if you are running the project locally, you will need to install the following libraries:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `seaborn`: For data visualization.
- `plotly`: For interactive visualizations.
- `matplotlib`: For static plotting.
- `scikit-learn`: For machine learning algorithms.
- `flask`: (optional) For deploying the model as a web application.

### To install these dependencies locally, use the following command:

```bash
pip install pandas numpy seaborn plotly matplotlib scikit-learn flask



Dataset
The dataset used in this project contains water quality data. The target variable is Potability, which indicates whether the water is potable (1) or not potable (0). The dataset consists of the following columns:

pH: pH level of the water.
Sulfate: Concentration of sulfate in the water.
Trihalomethanes: Concentration of trihalomethanes.
Potability: The target variable (1 = potable, 0 = not potable).
Dataset Sources:
The dataset is publicly available and can be loaded from your local machine or Google Drive.
If you're running this project in Google Colab, you can upload the dataset using:
python
Copy code


from google.colab import files
uploaded = files.upload()  # This will allow you to upload the dataset


Sure! Here’s a full README for your Water Potability Prediction machine learning project. I've included a description of the project, dependencies, instructions on how to run the notebook, and more. You can just copy-paste this into your README file.

Water Potability Prediction
Description
This machine learning project aims to predict whether water is potable (safe for drinking) or not based on various water quality features. The dataset contains information about the levels of different water quality parameters such as pH, Sulfate, Trihalomethanes, and other chemical compounds that impact water potability. By applying data preprocessing techniques, feature scaling, and exploring various machine learning models, this project demonstrates how different algorithms can be used to classify water as potable or non-potable.

The notebook is implemented in Google Colab, which makes it easy to run without any local setup. The project includes the following steps:

### Exploratory Data Analysis (EDA): Understanding the dataset, visualizing missing values, and exploring correlations between features.



Data Preprocessing: Handling missing values, scaling features, and preparing the dataset for training.
Model Training: Training multiple machine learning models, including:
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Model Evaluation: Evaluating model performance using accuracy scores and confusion matrices.
The project uses several popular Python libraries such as pandas, matplotlib, scikit-learn, and seaborn to handle the data and train the models.

Table of Contents
Description
How to Use
Dependencies
Dataset
Usage
Model Accuracy Comparison
License
How to Use
1. Open the Project in Google Colab:
Click the link below to open the project directly in Google Colab:

Open in Google Colab

2. Run the Notebook:
Once opened in Google Colab, run each cell by clicking the play button on the left side of each code block.
The notebook will guide you through the steps, from loading the dataset to training the machine learning models.
Dependencies
All required libraries are pre-installed in Google Colab. However, if you're running the project locally, you will need to install the following libraries:

pandas: For data manipulation and analysis.
numpy: For numerical computations.
seaborn: For data visualization.
plotly: For interactive visualizations.
matplotlib: For static plotting.
scikit-learn: For machine learning algorithms.
flask: (optional) For deploying the model as a web application.
To install these dependencies locally, use the following command:
bash
Copy code
pip install pandas numpy seaborn plotly matplotlib scikit-learn flask
Dataset
The dataset used in this project contains water quality data. The target variable is Potability, which indicates whether the water is potable (1) or not potable (0). The dataset consists of the following columns:

pH: pH level of the water.
Sulfate: Concentration of sulfate in the water.
Trihalomethanes: Concentration of trihalomethanes.
Potability: The target variable (1 = potable, 0 = not potable).
Dataset Sources:
The dataset is publicly available and can be loaded from your local machine or Google Drive.
If you're running this project in Google Colab, you can upload the dataset using:
python
Copy code
from google.colab import files
uploaded = files.upload()  # This will allow you to upload the dataset
Usage
1. Data Preprocessing & Exploratory Data Analysis (EDA)
In this step, we load the dataset, check for missing values, visualize correlations, and handle missing data by filling it with the mean of the column. The missing values in the features pH, Sulfate, and Trihalomethanes are filled with the column mean.
# Water Quality Prediction

## Description

This machine learning project predicts whether water is potable based on various features, such as pH level, Sulfate content, and Trihalomethanes concentration. The dataset is analyzed using exploratory data analysis (EDA), followed by preprocessing, model training, and evaluation using five different machine learning algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- KNN (K-Nearest Neighbors)
- SVM (Support Vector Machine)

The notebook is implemented in **Google Colab**, which provides an easy way to run Python code in the cloud without any installation required.

## Table of Contents

1. ## Description

This project uses machine learning to predict the potability of water based on various chemical and physical features. The dataset includes water quality parameters such as **pH**, **Sulfate**, and **Trihalomethanes**, and the goal is to classify water as either potable or non-potable. Using several machine learning algorithms, including Logistic Regression, Decision Trees, and Random Forest, we train and evaluate models to predict water potability.

2. [Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)



## How to Use

### 1. **Open the Project in Google Colab**:
Click the link below to open the project directly in **Google Colab**:

[Open in Google Colab](https://colab.research.google.com/github/your-username/your-repo/blob/main/your-notebook.ipynb)

### 2. **Run the Notebook**:
- Once opened in **Google Colab**, run each cell by clicking the **play button** on the left side of each code block.
- The notebook will guide you through the steps, from loading the dataset to training the machine learning models.

---

## Dependencies

All required libraries are pre-installed in **Google Colab**. However, if you are running the project locally, you will need to install the following libraries:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `seaborn`: For data visualization.
- `plotly`: For interactive visualizations.
- `matplotlib`: For static plotting.
- `scikit-learn`: For machine learning algorithms.
- `flask`: (optional) For deploying the model as a web application.

### To install these dependencies locally, use the following command:

```bash
pip install pandas numpy seaborn plotly matplotlib scikit-learn flask



Dataset
The dataset used in this project contains water quality data. The target variable is Potability, which indicates whether the water is potable (1) or not potable (0). The dataset consists of the following columns:

pH: pH level of the water.
Sulfate: Concentration of sulfate in the water.
Trihalomethanes: Concentration of trihalomethanes.
Potability: The target variable (1 = potable, 0 = not potable).
Dataset Sources:
The dataset is publicly available and can be loaded from your local machine or Google Drive.
If you're running this project in Google Colab, you can upload the dataset using:
python
Copy code


from google.colab import files
uploaded = files.upload()  # This will allow you to upload the dataset


Sure! Here’s a full README for your Water Potability Prediction machine learning project. I've included a description of the project, dependencies, instructions on how to run the notebook, and more. You can just copy-paste this into your README file.

Water Quality Prediction
Description
This machine learning project aims to predict whether water is potable (safe for drinking) or not based on various water quality features. The dataset contains information about the levels of different water quality parameters such as pH, Sulfate, Trihalomethanes, and other chemical compounds that impact water potability. By applying data preprocessing techniques, feature scaling, and exploring various machine learning models, this project demonstrates how different algorithms can be used to classify water as potable or non-potable.

The notebook is implemented in Google Colab, which makes it easy to run without any local setup. The project includes the following steps:

## Exploratory Data Analysis (EDA): Understanding the dataset, visualizing missing values, and exploring correlations between features.



Data Preprocessing: Handling missing values, scaling features, and preparing the dataset for training.
Model Training: Training multiple machine learning models, including:
Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Model Evaluation: Evaluating model performance using accuracy scores and confusion matrices.
The project uses several popular Python libraries such as pandas, matplotlib, scikit-learn, and seaborn to handle the data and train the models.

Table of Contents
Description
How to Use
Dependencies
Dataset
Usage
Model Accuracy Comparison
License
How to Use
1. Open the Project in Google Colab:
Click the link below to open the project directly in Google Colab:

Open in Google Colab

2. Run the Notebook:
Once opened in Google Colab, run each cell by clicking the play button on the left side of each code block.
The notebook will guide you through the steps, from loading the dataset to training the machine learning models.
Dependencies
All required libraries are pre-installed in Google Colab. However, if you're running the project locally, you will need to install the following libraries:

pandas: For data manipulation and analysis.
numpy: For numerical computations.
seaborn: For data visualization.
plotly: For interactive visualizations.
matplotlib: For static plotting.
scikit-learn: For machine learning algorithms.
flask: (optional) For deploying the model as a web application.
To install these dependencies locally, use the following command:
bash
Copy code
pip install pandas numpy seaborn plotly matplotlib scikit-learn flask
Dataset
The dataset used in this project contains water quality data. The target variable is Potability, which indicates whether the water is potable (1) or not potable (0). The dataset consists of the following columns:

pH: pH level of the water.
Sulfate: Concentration of sulfate in the water.
Trihalomethanes: Concentration of trihalomethanes.
Potability: The target variable (1 = potable, 0 = not potable).
Dataset Sources:
The dataset is publicly available and can be loaded from your local machine or Google Drive.
If you're running this project in Google Colab, you can upload the dataset using:
python
Copy code
from google.colab import files
uploaded = files.upload()  # This will allow you to upload the dataset
Usage
1. Data Preprocessing & Exploratory Data Analysis (EDA)
In this step, we load the dataset, check for missing values, visualize correlations, and handle missing data by filling it with the mean of the column. The missing values in the features pH, Sulfate, and Trihalomethanes are filled with the column mean.



🚀 Quick Implementation
1. Load Required Libraries


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# Handle missing values by filling with mean
df['ph'] = df['ph'].fillna(df['ph'].mean())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# Handle missing values by filling with mean
df['ph'] = df['ph'].fillna(df['ph'].mean())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean())




from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df.drop('Potability', axis=1)
y = df['Potability']

# Scale features
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, pred_lr) * 100:.2f}%")

# Decision Tree
model_dt = DecisionTreeClassifier(max_depth=4)
model_dt.fit(X_train, y_train)
pred_dt = model_dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, pred_dt) * 100:.2f}%")

# Random Forest
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, pred_rf) * 100:.2f}%")

# KNN
model_knn = KNeighborsClassifier(n_neighbors=22)
model_knn.fit(X_train, y_train)
pred_knn = model_knn.predict(X_test)
print(f"KNN Accuracy: {accuracy_score(y_test, pred_knn) * 100:.2f}%")

# SVM
model_svm = SVC(kernel='rbf')
model_svm.fit(X_train, y_train)
pred_svm = model_svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, pred_svm) * 100:.2f}%")
import matplotlib.pyplot as plt



🔥 Key Features & Results
# Model names and accuracy values
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'SVM']
accuracies = [
    accuracy_score(y_test, pred_lr) * 100,
    accuracy_score(y_test, pred_dt) * 100,
    accuracy_score(y_test, pred_rf) * 100,
    accuracy_score(y_test, pred_knn) * 100,
    accuracy_score(y_test, pred_svm) * 100
]
📊 Interactive Visualization


# Plot the accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color='blue')
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.show()
