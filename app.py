
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Titanic EDA",
    page_icon=None,
    layout="wide"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")

    # Data cleaning performed in the Jupyter Notebook
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Cabin"] = df["Cabin"].fillna(df["Cabin"].mode()[0])

    return df


df = load_data()


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("Titanic EDA")

page = st.sidebar.radio(
    "Select Page",
    [
        "Introduction",
        "Report",
        "Conclusion"
    ]
)


# =========================================================
# INTRODUCTION PAGE
# =========================================================
if page == "Introduction":

    st.title("Titanic Dataset Analysis")

    st.write(
        """
        This project presents an Exploratory Data Analysis (EDA) of the
        Titanic dataset. The analysis is based on passenger information
        such as passenger class, sex, age, number of siblings or spouses,
        parents or children, ticket information, fare, cabin and port of
        embarkation.
        """
    )

    st.header("Objective")

    st.write(
        """
        The main objective of this project is to understand the Titanic
        dataset, identify missing values, perform basic data cleaning,
        examine the structure of the data and visualize important
        characteristics of the passengers.
        """
    )

    st.header("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Passengers", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    with col3:
        st.metric("Survivors", int(df["Survived"].sum()))

    st.header("Columns Used")

    columns = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(columns, use_container_width=True)

    st.header("Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)


# =========================================================
# REPORT PAGE
# =========================================================
elif page == "Report":

    st.title("Titanic Dataset Report")

    # -----------------------------------------------------
    # Dataset Overview
    # -----------------------------------------------------
    st.header("1. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Average Age", f"{df['Age'].mean():.2f}")

    with col4:
        st.metric("Average Fare", f"{df['Fare'].mean():.2f}")

    st.write(
        """
        The dataset contains 891 passenger records and 12 columns.
        The columns contain numerical and categorical information about
        Titanic passengers.
        """
    )

    # -----------------------------------------------------
    # Data Types
    # -----------------------------------------------------
    st.header("2. Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(dtype_df, use_container_width=True)

    # -----------------------------------------------------
    # Missing Values
    # -----------------------------------------------------
    st.header("3. Missing Values")

    original_df = pd.read_csv("Titanic-Dataset.csv")

    missing_values = original_df.isnull().sum()
    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(missing_df, use_container_width=True)

    st.write(
        """
        In the original dataset, Age, Cabin and Embarked contain missing
        values. The notebook handles these missing values by replacing
        Age with its median and replacing categorical missing values with
        their mode.
        """
    )

    # -----------------------------------------------------
    # Descriptive Statistics
    # -----------------------------------------------------
    st.header("4. Descriptive Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    # -----------------------------------------------------
    # Passenger Class
    # -----------------------------------------------------
    st.header("5. Passenger Class Distribution")

    class_counts = df["Pclass"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        class_counts.index.astype(str),
        class_counts.values
    )

    ax.bar_label(bars)

    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Number of Passengers")
    ax.set_title("Passenger Distribution by Class")

    st.pyplot(fig)

    # -----------------------------------------------------
    # Survival Analysis
    # -----------------------------------------------------
    st.header("6. Survival Analysis")

    survival_counts = df["Survived"].value_counts().sort_index()

    survival_labels = ["Did Not Survive", "Survived"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        survival_labels,
        survival_counts.values
    )

    ax.bar_label(bars)

    ax.set_xlabel("Survival Status")
    ax.set_ylabel("Number of Passengers")
    ax.set_title("Survival Distribution")

    st.pyplot(fig)

    # -----------------------------------------------------
    # Survival by Gender
    # -----------------------------------------------------
    st.header("7. Survival by Sex")

    survival_sex = pd.crosstab(
        df["Sex"],
        df["Survived"]
    )

    survival_sex.columns = [
        "Did Not Survive",
        "Survived"
    ]

    st.dataframe(
        survival_sex,
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    survival_sex.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Sex")
    ax.set_ylabel("Number of Passengers")
    ax.set_title("Survival Distribution by Sex")

    ax.tick_params(axis="x", rotation=0)

    st.pyplot(fig)

    # -----------------------------------------------------
    # Survival by Passenger Class
    # -----------------------------------------------------
    st.header("8. Survival by Passenger Class")

    class_survival = pd.crosstab(
        df["Pclass"],
        df["Survived"]
    )

    class_survival.columns = [
        "Did Not Survive",
        "Survived"
    ]

    st.dataframe(
        class_survival,
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    class_survival.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Number of Passengers")
    ax.set_title("Survival Distribution by Passenger Class")

    ax.tick_params(axis="x", rotation=0)

    st.pyplot(fig)

    # -----------------------------------------------------
    # Age Distribution
    # -----------------------------------------------------
    st.header("9. Age Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df["Age"],
        bins=20
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Passengers")
    ax.set_title("Age Distribution of Passengers")

    st.pyplot(fig)

    # -----------------------------------------------------
    # Fare Distribution
    # -----------------------------------------------------
    st.header("10. Fare Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df["Fare"],
        bins=20
    )

    ax.set_xlabel("Fare")
    ax.set_ylabel("Number of Passengers")
    ax.set_title("Fare Distribution")

    st.pyplot(fig)

    # -----------------------------------------------------
    # Cleaned Dataset
    # -----------------------------------------------------
    st.header("11. Cleaned Dataset")

    st.write(
        "After handling the missing values, the cleaned dataset contains:"
    )

    cleaned_missing = df.isnull().sum()

    cleaned_missing_df = pd.DataFrame({
        "Column": cleaned_missing.index,
        "Remaining Missing Values": cleaned_missing.values
    })

    st.dataframe(
        cleaned_missing_df,
        use_container_width=True
    )

    st.header("12. Cleaned Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# CONCLUSION PAGE
# =========================================================
elif page == "Conclusion":

    st.title("Conclusion")

    st.write(
        """
        The Titanic dataset was explored using Python, Pandas and
        Matplotlib. The analysis included understanding the dataset
        structure, checking data types, identifying missing values,
        handling missing values and performing descriptive analysis.
        """
    )

    st.header("Key Findings")

    st.write(
        """
        1. The dataset contains 891 passenger records and 12 columns.

        2. The original dataset contains missing values in Age, Cabin
           and Embarked.

        3. Missing Age values were replaced using the median Age.

        4. Missing categorical values in Embarked and Cabin were handled
           using the mode.

        5. Passenger class distribution was examined using a bar chart.

        6. Survival distribution was analyzed using passenger survival
           status.

        7. Survival was also examined according to passenger sex and
           passenger class.

        8. Age and Fare distributions were visualized to understand
           their overall patterns.
        """
    )

    st.header("Final Summary")

    st.write(
        """
        The exploratory data analysis provides an overview of the
        Titanic passengers and demonstrates the complete basic EDA
        workflow: loading data, inspecting data, identifying missing
        values, cleaning data, calculating descriptive statistics and
        creating visualizations.
        """
    )

    st.success(
        "Titanic Exploratory Data Analysis completed successfully."
    )

