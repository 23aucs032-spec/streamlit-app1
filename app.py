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

# NEW IMPORT for dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Streamlit App", layout="wide")

# ---------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------
st.sidebar.title("Model Type")

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

    test_size_display = st.sidebar.slider("Test Size (10 - 50%)", 10, 50)
    test_size = test_size_display / 100

elif model_type == "Classification":
    algorithm = st.sidebar.selectbox(
        "Algorithms",
        ["Logistic Regression", "Random Forest Classifier", "Support Vector Classifier",
         "Decision Tree Classifier", "KNN Classifier"]
    )

    test_size_display = st.sidebar.slider("Test Size (10 - 50%)", 10, 50)
    test_size = test_size_display / 100

else:
    algorithm = st.sidebar.selectbox(
        "Algorithms",
        ["K-Means Clustering", "Agglomerative Clustering"]
    )
    test_size = None

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

    # Submit Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_clicked = st.button("Submit")

    if submit_clicked:

        if len(feature_cols) < 1:
            st.error("You must select at least one feature!")
            st.stop()

        if model_type != "Clustering" and label_col in feature_cols:
            st.error("Label column cannot be selected as a feature!")
            st.stop()

        X = df[feature_cols]

        # ===========================================
        # CLUSTERING
        # ===========================================
        if model_type == "Clustering":

            # --- K-Means ---
            if algorithm == "K-Means Clustering":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(X)
                df["Cluster"] = clusters

                st.success("K-Means Clustering Completed Successfully!")
                st.write(f"**Number of Clusters:** {n_clusters}")

                if len(feature_cols) >= 2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
                    ax.set_xlabel(feature_cols[0])
                    ax.set_ylabel(feature_cols[1])
                    ax.set_title(f"K-Means Cluster Visualization ({n_clusters} clusters)")
                    st.pyplot(fig)
                else:
                    st.warning("Select at least 2 features to show clustering graph!")

            # --- Agglomerative Clustering ---
            elif algorithm == "Agglomerative Clustering":
                st.success("Agglomerative Clustering Completed Successfully!")
                st.write(f"**Number of Clusters:** {n_clusters}")

                # Fit Model
                model = AgglomerativeClustering(n_clusters=n_clusters)
                clusters = model.fit_predict(X)
                df["Cluster"] = clusters

                # ---------------------------
                # DENDROGRAM SECTION
                # ---------------------------
                st.write("### Dendrogram (Hierarchical Tree)")

                linked = linkage(X, method='ward')

                fig, ax = plt.subplots(figsize=(10, 6))
                dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=True)
                ax.set_title("Agglomerative Clustering Dendrogram")
                ax.set_xlabel("Samples")
                ax.set_ylabel("Distance")
                st.pyplot(fig)

                # Scatter plot (if 2 features)
                if len(feature_cols) >= 2:
                    st.write("### Cluster Visualization")
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
                    ax2.set_xlabel(feature_cols[0])
                    ax2.set_ylabel(feature_cols[1])
                    ax2.set_title("Agglomerative Clustering Visualization")
                    st.pyplot(fig2)
                else:
                    st.warning("Select at least 2 features to show cluster graph!")

        # ===========================================
        # REGRESSION
        # ===========================================
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
            elif algorithm == "KNN Regressor":
                model = KNeighborsRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("Regression Model Trained Successfully!")
            st.write(f"MSE: {mean_squared_error(y_test, y_pred)}")
            st.write(f"R² Score: {r2_score(y_test, y_pred)}")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

        # ===========================================
        # CLASSIFICATION
        # ===========================================
        else:

            y = df[label_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

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
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
