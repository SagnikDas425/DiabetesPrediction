from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    data = pd.read_csv("D:\\DiabetesPredicition\\diabetes_prediction_dataset.csv")

    # Data preprocessing
    data['smoking_history'].replace(['never', 'No Info', 'current', 'not current', 'former', 'ever'],
                                    [0, 0, 1, 0, 1, 1], inplace=True)
    data['gender'].replace(['Female', 'Male', 'Other'], [0, 1, 0], inplace=True)

    # Check for missing values
    data = data.dropna()

    # Define features and target
    x = data.drop("diabetes", axis=1)
    y = data['diabetes']

    # Standardize the features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Handle data imbalance using SMOTE
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    # Model training
    model = LogisticRegression()
    model.fit(x_train_res, y_train_res)

    # Evaluate model
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    threshold = 0.3  # Adjust based on precision-recall curve
    y_pred_adjusted = (y_pred_prob >= threshold).astype(int)

    print(f"Accuracy: {accuracy_score(y_test, y_pred_adjusted)}")
    print(f"Precision: {precision_score(y_test, y_pred_adjusted)}")
    print(f"Recall: {recall_score(y_test, y_pred_adjusted)}")
    print(f"F1 Score: {f1_score(y_test, y_pred_adjusted)}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred_adjusted)}")

    # Get input values from the request
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    # Standardize the input values
    input_data = np.array([[val1, val2, val3, val4, val5, val6, val7, val8]])
    input_data = scaler.transform(input_data)

    # Make prediction
    input_pred_prob = model.predict_proba(input_data)[:, 1]
    input_pred_adjusted = (input_pred_prob >= threshold).astype(int)

    result1 = "You are Diabetes Positive" if input_pred_adjusted == 1 else "You are Diabetes Negative"

    return render(request, "predict.html", {"result2": result1})
