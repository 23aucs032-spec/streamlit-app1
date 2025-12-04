import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Streamlit ML App", layout="wide")

# ----------------------------------------------------
# SIDEBAR OPTIONS
# ----------------------------------------------------
st.sidebar.title("Model Type")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Regression", "Classification", "Clustering"]
)

# Regression Options
if model_type == "Regression":
    algorithm = st.sidebar.selectbox(
        "Regression Algorithm",
        ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor",
         "Decision Tree Regressor", "KNN Regressor"]
    )
    test_size = st.sidebar.slider("Test Size (%)", 10, 50) / 100

    if algorithm in ["Decision Tree Regressor", "Random Forest Regressor"]:
        max_depth = st.sidebar.number_input("Max Depth", 1, 50, 5)

# Classification Options
elif model_type == "Classification":
    algorithm = st.sidebar.selectbox(
        "Classification Algorithm",
        ["Logistic Regression", "Random Forest Classifier", "Support Vector Classifier",
         "Decision Tree Classifier", "KNN Classifier"]
    )
    test_size = st.sidebar.slider("Test Size (%)", 10, 50) / 100

    if algorithm in ["Decision Tree Classifier", "Random Forest Classifier"]:
        max_depth = st.sidebar.number_input("Max Depth", 1, 50, 5)

# Clustering Options
else:
    algorithm = st.sidebar.selectbox(
        "Clustering Algorithm",
        ["K-Means Clustering", "Agglomerative Clustering"]
    )
    test_size = None

# ----------------------------------------------------
# MAIN UI
# ----------------------------------------------------
st.title("Machine Learning Platform")
st.subheader(f"{algorithm}")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ----------------------------------------------------
# FILE PROCESSING
# ----------------------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df = df.drop(columns=[c for c in ["Name", "Cabin", "Ticket"] if c in df.columns], errors="ignore")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    labelencoder = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = labelencoder.fit_transform(df[col].astype(str))

    st.write("Processed Data")
    st.dataframe(df)

    feature_cols = st.multiselect("Select Feature Columns (X)", df.columns)

    if model_type != "Clustering":
        label_col = st.selectbox("Select Label Column (Y)", df.columns)
    else:
        label_col = None
        n_clusters = st.number_input("Clusters", 1, 20, 3)

    # ---------- SUBMIT BUTTON CENTERED ----------
    col1, col2, col3 = st.columns([3,2,3])
    with col2:
        run_model = st.button("Submit")
    # --------------------------------------------

    if run_model:

        if len(feature_cols) < 1:
            st.error("Please select at least one feature column.")
            st.stop()

        if model_type != "Clustering" and label_col in feature_cols:
            st.error("Label column cannot be a feature.")
            st.stop()

        X = df[feature_cols]

        # -------------------- CLUSTERING --------------------
        if model_type == "Clustering":

            if algorithm == "K-Means Clustering":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            else:
                model = AgglomerativeClustering(n_clusters=n_clusters)

            clusters = model.fit_predict(X)
            df["Cluster"] = clusters

            st.success("Clustering Completed!")
            st.dataframe(df[["Cluster"]])

            if len(feature_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 features to plot clusters.")

        # -------------------- REGRESSION --------------------
        elif model_type == "Regression":

            y = df[label_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "Random Forest Regressor":
                model = RandomForestRegressor(max_depth=max_depth, random_state=42)
            elif algorithm == "Support Vector Regressor":
                model = SVR()
            elif algorithm == "Decision Tree Regressor":
                model = DecisionTreeRegressor(max_depth=max_depth)
            else:
                model = KNeighborsRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success("Regression Completed!")
            st.write(f"### MSE: {mse}")
            st.write(f"### R² Score: {r2}")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred)
            st.pyplot(fig)

            # Tree visualizations
            if algorithm == "Decision Tree Regressor":
                st.subheader("Decision Tree Visualization")
                fig = plt.figure(figsize=(20,12))
                plot_tree(model, filled=True, feature_names=feature_cols)
                st.pyplot(fig)

            elif algorithm == "Random Forest Regressor":
                st.subheader("Random Forest First Tree Visualization")
                tree_model = model.estimators_[0]
                fig = plt.figure(figsize=(20,12))
                plot_tree(tree_model, filled=True, feature_names=feature_cols)
                st.pyplot(fig)

        # -------------------- CLASSIFICATION --------------------
        else:

            y = df[label_col]

            if y.dtype in ["int64", "float64"] and y.nunique() > 10:
                y = pd.qcut(y, q=4, labels=[0, 1, 2, 3])
                y = y.astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=500)
            elif algorithm == "Random Forest Classifier":
                model = RandomForestClassifier(max_depth=max_depth, random_state=42)
            elif algorithm == "Support Vector Classifier":
                model = SVC()
            elif algorithm == "Decision Tree Classifier":
                model = DecisionTreeClassifier(max_depth=max_depth)
            else:
                model = KNeighborsClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.success("Classification Completed!")
            st.write(f"### Accuracy: {accuracy}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
            st.pyplot(fig)

            if algorithm == "Decision Tree Classifier":
                st.subheader("Decision Tree Visualization")
                fig = plt.figure(figsize=(20,12))
                plot_tree(model, filled=True, feature_names=feature_cols, class_names=True)
                st.pyplot(fig)

            elif algorithm == "Random Forest Classifier":
                st.subheader("Random Forest First Tree Visualization")
                tree_model = model.estimators_[0]
                fig = plt.figure(figsize=(20,12))
                plot_tree(tree_model, filled=True, feature_names=feature_cols, class_names=True)
                st.pyplot(fig)
