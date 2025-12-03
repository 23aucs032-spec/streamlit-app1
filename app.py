import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Streamlit App", layout="wide")

# ---------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------
st.sidebar.title("Model Type")

# --- Model Type ---
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Regression", "Classification", "Clustering"]
)

# --- Algorithm Selection ---
if model_type == "Regression":
    algorithm = st.sidebar.selectbox(
        "Algorithms",
        ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor",
         "Decision Tree Regressor", "KNN Regressor"]
    )

    # Train/Test Size Slider for Regression only (10 - 50)
    test_size_display = st.sidebar.slider(
        "Test Size (10 - 50%)", 10, 50, step=1
    )
    test_size = test_size_display / 100  # convert to fraction for train_test_split

elif model_type == "Classification":
    algorithm = st.sidebar.selectbox(
        "Algorithms",
        ["Logistic Regression", "Random Forest Classifier", "Support Vector Classifier",
         "Decision Tree Classifier", "KNN Classifier"]
    )

    # Train/Test Size Slider for Classification only (10 - 50)
    test_size_display = st.sidebar.slider(
        "Test Size (10 - 50%)", 10, 50, step=1
    )
    test_size = test_size_display / 100  # convert to fraction for train_test_split

else:
    algorithm = st.sidebar.selectbox(
        "Algorithms",
        ["K-Means Clustering", "Agglomerative Clustering"]
    )
    test_size = None  # Clustering doesn't use train/test split

# ---------------------------------------------
# MAIN PAGE HEADING
# ---------------------------------------------
st.title("Machine Learning Platform")
st.subheader(f"{algorithm}")

# ---------------------------------------------
# File Upload
# ---------------------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df)
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    feature_cols = st.multiselect("Select Features", df.columns)

    if model_type != "Clustering":
        label_col = st.selectbox("Select Label Column", df.columns)
        n_clusters = None
    else:
        n_clusters = st.number_input("Select Number of Clusters", min_value=1, max_value=20, value=3)
        label_col = None

    # ---------------------------------------------
    # CENTERED SUBMIT BUTTON
    # ---------------------------------------------
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        submit_clicked = st.button("Submit")

    if submit_clicked:
        if len(feature_cols) < 1:
            st.error("You must select at least one feature!")
            st.stop()
        if model_type != "Clustering" and label_col in feature_cols:
            st.error("Label column cannot be a feature!")
            st.stop()

        X = df[feature_cols]

        # ============================================================
        # CLUSTERING
        # ============================================================
        if model_type == "Clustering":
            if algorithm == "K-Means Clustering":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif algorithm == "Agglomerative Clustering":
                model = AgglomerativeClustering(n_clusters=n_clusters)

            clusters = model.fit_predict(X)
            df["Cluster"] = clusters  # stored but not displayed

            st.success("Clustering Completed Successfully!")

            # Show number of clusters
            st.write(f"**Number of Clusters:** {n_clusters}")

            if len(feature_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
                ax.set_xlabel(feature_cols[0])
                ax.set_ylabel(feature_cols[1])
                ax.set_title(f"Cluster Visualization ({n_clusters} clusters)")
                st.pyplot(fig)
            else:
                st.warning("Select at least 2 features to show the clustering graph!")

        # ============================================================
        # REGRESSION
        # ============================================================
        elif model_type == "Regression":
            y = df[label_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if algorithm == "Linear Regression":
                model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_x)
            elif algorithm == "Random Forest Regressor":
                model = RandomForestRegressor()
            elif algorithm == "Support Vector Regressor":
                model = SVR()
            elif algorithm == "Decision Tree Regressor":
                model = DecisionTreeRegressor()
            elif algorithm == "KNN Regressor":
                model = KNeighborsRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.success("Regression Model Trained Successfully!")

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"MSE: {mse}")
            st.write(f"R² Score: {r2}")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

        # ============================================================
        # CLASSIFICATION
        # ============================================================
        else:
            y = df[label_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if algorithm == "Logistic Regression":
                model = LogisticRegression()
            elif algorithm == "Random Forest Classifier":
                model = RandomForestClassifier()
            elif algorithm == "Support Vector Classifier":
                model = SVC()
            elif algorithm == "Decision Tree Classifier":
                model = DecisionTreeClassifier()
            elif algorithm == "KNN Classifier":
                model = KNeighborsClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.success("Classification Model Trained Successfully!")

            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
