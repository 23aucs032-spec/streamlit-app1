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

from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Machine Learning App", layout="wide")

# ---------------------------------------------
# SIDEBAR
# ---------------------------------------------
st.sidebar.title("Model Type")

model_type = st.sidebar.selectbox(
    "Choose Model Type",
    ["Regression", "Classification", "Clustering"]
)

# Algorithms
if model_type == "Regression":
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor",
         "Decision Tree Regressor", "KNN Regressor"]
    )
    test_size_display = st.sidebar.slider("Test Size (10% - 50%)", 10, 50)
    test_size = test_size_display / 100

elif model_type == "Classification":
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["Logistic Regression", "Random Forest Classifier", "Support Vector Classifier",
         "Decision Tree Classifier", "KNN Classifier"]
    )
    test_size_display = st.sidebar.slider("Test Size (10% - 50%)", 10, 50)
    test_size = test_size_display / 100

else:
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["K-Means Clustering", "Agglomerative Clustering"]
    )
    test_size = None

# ---------------------------------------------
# MAIN PAGE
# ---------------------------------------------
st.title("Machine Learning Platform")
st.subheader(f"Selected Algorithm: {algorithm}")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # CLEANING: Drop columns that cause ML issues
    df = df.drop(columns=[c for c in ["Name", "Cabin", "Ticket"] if c in df.columns], errors='ignore')

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Label Encoding for categorical columns
    labelencoder = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = labelencoder.fit_transform(df[col].astype(str))

    st.write("### Processed Data")
    st.dataframe(df)

    feature_cols = st.multiselect("Select Features (X)", df.columns)

    if model_type != "Clustering":
        label_col = st.selectbox("Select Label (Y)", df.columns)
        n_clusters = None
    else:
        label_col = None
        n_clusters = st.number_input("Number of Clusters", min_value=1, max_value=20, value=3)

    # Submit button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        submit_clicked = st.button("Run Model")

    if submit_clicked:

        if len(feature_cols) < 1:
            st.error("Select at least one feature column.")
            st.stop()

        if model_type != "Clustering" and label_col in feature_cols:
            st.error("Label column cannot be in features.")
            st.stop()

        X = df[feature_cols]

        # ============================================================
        # CLUSTERING
        # ============================================================
        if model_type == "Clustering":

            if algorithm == "K-Means Clustering":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            else:
                model = AgglomerativeClustering(n_clusters=n_clusters)

            clusters = model.fit_predict(X)
            df["Cluster"] = clusters

            st.success("Clustering Completed Successfully!")

            if len(feature_cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
                ax.set_xlabel(feature_cols[0])
                ax.set_ylabel(feature_cols[1])
                ax.set_title(f"Clustering ({n_clusters} clusters)")
                st.pyplot(fig)
            else:
                st.warning("Select at least 2 features to show clustering plot.")

        # ============================================================
        # REGRESSION
        # ============================================================
        elif model_type == "Regression":

            y = df[label_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            if algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "Random Forest Regressor":
                model = RandomForestRegressor()
            elif algorithm == "Support Vector Regressor":
                model = SVR()
            elif algorithm == "Decision Tree Regressor":
                model = DecisionTreeRegressor()
            else:
                model = KNeighborsRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success("Regression Model Trained Successfully!")
            st.write(f"### MSE: {mse}")
            st.write(f"### R² Score: {r2}")

            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

        # ============================================================
        # CLASSIFICATION (WITH FIX FOR CONTINUOUS LABELS)
        # ============================================================
        else:

            y = df[label_col]

            # ❗ FIX: Prevent Classification on continuous numeric labels
            if y.dtype in ['int64', 'float64'] and y.nunique() > 20:
                st.error("❌ Invalid Label for Classification.\n"
                         "This label is continuous. Please switch to Regression.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            if algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=300)
            elif algorithm == "Random Forest Classifier":
                model = RandomForestClassifier()
            elif algorithm == "Support Vector Classifier":
                model = SVC()
            elif algorithm == "Decision Tree Classifier":
                model = DecisionTreeClassifier()
            else:
                model = KNeighborsClassifier()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success("Classification Model Trained Successfully!")
            st.write(f"### Accuracy: {acc}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
