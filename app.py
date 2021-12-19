# Deploy de Aplicações Preditivas com Streamlit

# Imports
import time

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Slide Bar
st.write("** Deploy Machine Learning Models **")
st.write("** Logistic Regression **")

# SideBar
st.sidebar.header("Dataset end Hiperparameters")
st.sidebar.markdown("""**Select the Dataset**""")
Dataset = st.sidebar.selectbox("Dataset", ("Iris", "Wine", "Breast Cancer"))
Split = st.sidebar.slider("Select the Percent of Data train/test for the Model - (default = 70/30)", 0.2, 0.8, 0.70)
st.sidebar.markdown("""**Select the Hiperparameters**""")
Solver = st.sidebar.selectbox("Algorithm", ("lbfgs", "newton-cg", "liblinear", "sag"))
Penality = st.sidebar.radio("Regularization:", ("none", "l1", "l2", "elasticnet"))
# Tol = st.sidebar.text_input("Tolerance Stopping Criteria (default = 1e-4):", "4")
Max_Iteration = st.sidebar.text_input("Iterations Number (default = 50):", "50")

# Dicionário Para os Hiperparâmetros
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

parameters = {"Penality": Penality, "Max_Iteration": Max_Iteration, "Solver": Solver}
# Functions to Prepare the Data
def load_dataset(dataset):
    df = pd.DataFrame()
    if dataset == "Iris":
        df = sklearn.datasets.load_iris()
    elif dataset == "Wine":
        df = sklearn.datasets.load_wine()
    elif dataset == "Breast Cancer":
        df = sklearn.datasets.load_breast_cancer()
    return df


# Prepare the Data train/test
def prepare_data(df, split):
    X_treino, X_test, y_treino, y_test = train_test_split(df.data, df.target, test_size=float(split), random_state=42)

    # Scaler to Padronize the dataset
    scaler = MinMaxScaler()

    # Fit and Tranform train data
    X_treino = scaler.fit_transform(X_treino)

    # Tranform test data
    X_test = scaler.transform(X_test)

    return (X_treino, X_test, y_treino, y_test)


# Functions to MLeaning Model
def create_model(parameters):
    X_treino, X_test, y_treino, y_test = prepare_data(Data, Split)

    # Create the Model
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    clf = LogisticRegression(
        penalty=parameters["Penality"], solver=parameters["Solver"], max_iter=int(parameters["Max_Iteration"]),
    )

    clf = clf.fit(X_treino, y_treino)

    prediction = clf.predict(X_test)

    accuracy = sklearn.metrics.accuracy_score(y_test, prediction)

    cm = confusion_matrix(y_test, prediction)

    dict_value = {
        "Modelo": clf,
        "Accuracy": accuracy,
        "Prediction": prediction,
        "y_real": y_test,
        "Metrics": cm,
        "X_test": X_test,
    }

    return dict_value


# Web Applicartion Body
st.markdown("Resume")
st.write("Dataset Name:", Dataset)

Data = load_dataset(Dataset)

targets = Data.target_names

Dataframe = pd.DataFrame(Data.data, columns=Data.feature_names)
Dataframe["target"] = pd.Series(Data.target)
Dataframe["target labels"] = pd.Series(targets[i] for i in Data.target)

st.write("Attributes")
st.write(Dataframe)

# Button
if st.sidebar.button("Training the Logistic Regression Model"):
    with st.spinner("Loading the Dataset..."):
        time.sleep(1)

    st.success("Dataset Loaded")

    model = create_model(parameters)
    mybar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        mybar.progress(percent_complete + 1)

    with st.spinner("Training the Model..."):
        time.sleep(1)

    st.success("Data Loaded!")

    real_label = [targets[i] for i in model["y_real"]]

    predict_label = [targets[i] for i in model["Prediction"]]

    st.subheader("Predict the Model on Data Test")

    st.write(
        pd.DataFrame(
            {
                "Real Value: ": model["y_real"],
                "Real Label: ": real_label,
                "Real Predict: ": model["Prediction"],
                "Real Predict: ": predict_label,
            }
        )
    )

    matriz = model["Metrics"]
    st.subheader("Confusion Matrix")
    st.write(matriz)
    st.write("Accuracy: ", model["Accuracy"])
    st.write("Finished!!!")
