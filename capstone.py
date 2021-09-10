#Import necessary preprocessing libraries
import streamlit as st
import numpy as np
import pandas as pd
from pycorrcat.pycorrcat import plot_corr, corr_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler

#Import metrics libraries
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

#Import classification methods
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Set seed
seed=0

#Page title
def main():
    st.title("Predicting Financial Inclusion")
    st.markdown("### The case of Indonesia")
    #Page description
    st.markdown("This is a dashboard of visualisations from the Indonesian Family Life Survey dataset ." 
    "The dataset includes household assets, simple demographics, and wealth variables, from 2007. "
    "The target variable is the ownership of financial instruments in 2014.")
    st.sidebar.title("Build model")
    st.sidebar.markdown("Fine tune a model to make predictions.")
if __name__ == '__main__':
    main()

#Read raw data (after feature selection)
df = pd.read_csv("ifls_hh_reduced.csv")
df = pd.DataFrame(df).iloc[:,1:]
cat_cols = df.loc[:,df.nunique()<=30].columns
df[cat_cols] = df[cat_cols].astype(int)

#Sidebar parameters: Checkbox to display raw data or correlation heatmap
if st.sidebar.checkbox("Display data", False):
    st.subheader("Indonesian Family Life Survey Data (2007-2014)")
    st.dataframe(df)

if st.sidebar.checkbox("Display correlation heatmap", False):
    st.subheader("Correlation Heatmap - Cramer's V")
    st.image('visualisations/corr_heatmap.png')

#Get X and y
le = LabelEncoder()
X = df.drop(['own_fininstrument_14'], axis=1)
y = df['own_fininstrument_14']
y = le.fit_transform(y)

#Identify columns with more than 18 unique values (treat as numerical)
num = X.loc[:,X.nunique()>18].columns

#Identify columns with 18 unique values or less (categorical)
cat = X.loc[:,X.nunique()<=18].columns

encoder = OneHotEncoder(sparse=False, drop="if_binary")

#Preprocessing
X2 = encoder.fit_transform(X[cat])
X2 = pd.DataFrame(X2)
X3 = X[num].reset_index(drop=True)
X4 = pd.concat([X2, X3], axis=1, join='outer')

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X4, y)

#Standard Scaler
t = [(('scaler'), StandardScaler(), num)]
ct = ColumnTransformer(transformers=t, remainder='passthrough')

#Fit and transform training data
X_train = ct.fit_transform(X_train)
X_train = pd.DataFrame(X_train)

#Transform test data
X_test = ct.transform(X_test)

#Undersample
rus = RandomUnderSampler(random_state=seed)
X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

#Create Evaluation Metrics
def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        fig1 = plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
        st.pyplot(fig1.figure_)

    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        fig2 = plot_roc_curve(model, X_test, y_test)
        st.pyplot(fig2.figure_)

    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        fig3 = plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot(fig3.figure_)

class_names = ["Banked", "Unbanked"]

#Choose classifier
st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "Gradient Boosting"))

#Finetune SVM hyperparameters !! Remember to use undersampled dataset (X_train_us, y_train_us)
if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
    gamma = st.sidebar.radio("Gamma (Kernal coefficient)", ("scale", "auto"), key="gamma")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(X_train_us, y_train_us)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
        plot_metrics(metrics)


#Finetune Logistic Regression hyperparameters

if classifier == "Logistic Regression":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train_us, y_train_us)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

#Finetune Random Forest hyperparameters
if classifier == "Random Forest":
    st.sidebar.subheader("Hyperparameters")
    n_estimators= st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step =1, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap= bootstrap, n_jobs=-1 )
        model.fit(X_train_us, y_train_us)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

#Finetune Random Forest hyperparameters
if classifier == "Gradient Boosting":
    st.sidebar.subheader("Hyperparameters")
    n_estimators= st.sidebar.number_input("The number of boosting stages to perform", 100, 5000, step=10, key="n_estimators")
    min_samples_split= st.sidebar.number_input("The minimum number of samples in a node before splitting", 2, 100, step=1, key="min_samples_split")
    max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step =1, key="max_depth")
    loss = st.sidebar.radio("Loss function to be optimised", ("deviance", "exponential"), key="loss")
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Gradient Boosting Results")
        model = GradientBoostingClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth, loss=loss)
        model.fit(X_train_us, y_train_us)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)





