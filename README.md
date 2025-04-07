# Table of Contents

1.  [Project Objective](#project-objective)
2.  [Dataset](#dataset)
3.  [Key Tasks Performed](#key-tasks-performed)
4.  [Technologies Used](#technologies-used)
5.  [How to Use](#how-to-use)

# Retail Customer Spending Analysis and Prediction & LED signal noise clear

## Project Objective

This project performs a comprehensive Exploratory Data Analysis (EDA) on a retail customer dataset and develops a machine learning model to predict customer spending score categories. The analysis includes data cleaning, visualization, feature engineering, model training, evaluation, and incorporates advanced concepts like Bayesian priors and OOP encapsulation with unit testing. An additional signal processing task on noisy LED data is also included.

## Dataset

The primary dataset (`customer_data.csv`) contains information about retail store customers, including:

*   `Customer_ID`: Unique identifier for each customer.
*   `Age`: Age of the customer.
*   `Gender`: Gender of the customer ('Male' or 'Female').
*   `Income`: Annual income of the customer.
*   `Spending_Score`: A score assigned based on customer spending behavior (continuous).

An additional dataset (`light_sig_with_noise.csv`) was used for a separate signal processing bonus task.

## Key Tasks Performed

<details>
<summary>1. Data Loading and Cleaning</summary>

*   Loaded the dataset using Pandas.
*   Handled irrelevant columns (`Unnamed: 0`, `age` with sparse data).
*   Checked for and confirmed no duplicate `customer_id`s.
*   Imputed missing values: Median for numerical features (`incomes`, `spending_scores`), Mode for categorical (`genders`).
*   Converted `genders` to numerical representation using one-hot encoding.
*   Identified and handled outliers using standard deviation (for `incomes`) and IQR (for `spending_scores`).
</details>

<details>
<summary>2. Exploratory Data Analysis (EDA)</summary>

*   Visualized the distributions of `Age`, `Income`, and `Spending_Score` using histograms, KDE plots, violin plots, and bar charts (after binning where appropriate).
*   Visualized the `Gender` distribution using a pie chart.
*   Calculated and analyzed Pearson correlation coefficients between numerical features.
*   Visualized the relationship between `Income` and `Gender` using box plots and bar charts (with binned income).
</details>

<details>
<summary>3. Machine Learning - Classification</summary>

*   **Problem Conversion:** Transformed the continuous `Spending_Score` into a categorical target variable by binning the scores into 11 distinct categories (0-10).
*   **Data Preparation:** Split the data into training (80%) and testing (20%) sets. Implemented feature scaling (MinMaxScaler) on numerical features (`ages`, `incomes`).
*   **Model Exploration:** Trained and evaluated several classification algorithms:
    *   Gaussian Naive Bayes (GNB)
    *   K-Nearest Neighbors (KNN)
    *   Decision Tree (DT)
    *   Random Forest (RF)
*   **Hyperparameter Tuning:** Used `GridSearchCV` to find optimal hyperparameters for KNN (k neighbors) and Decision Trees (max_depth, min_samples_split), optimizing for both Accuracy and F1-score.
*   **Cross-Validation:** Employed 10-fold cross-validation during model evaluation and hyperparameter tuning to ensure robustness.
*   **Model Evaluation:** Assessed models using standard classification metrics: Accuracy, F1-Score (macro), Precision (macro), and Recall (macro).
*   **Model Selection:** Compared performance across models and selected the Decision Tree model (with specific hyperparameters) based on the Accuracy metric, considering the balanced nature of the created classes.
</details>

<details>
<summary>4. Bonus 1: Bayesian Statistics</summary>

*   Calculated prior probabilities for the spending score categories.
*   Attempted to integrate these priors into the Gaussian Naive Bayes model to potentially improve predictions (results indicated no significant improvement in this specific case).
</details>

<details>
<summary>5. Bonus 2: OOP Encapsulation and Unit Testing</summary>

*   Encapsulated the data splitting, model training (GNB, KNN, DT, RF), hyperparameter search, and evaluation logic within a Python class (`store_customers_data_science_class`).
*   Implemented unit tests using the `unittest` library to verify the functionality of data splitting, model fitting, and evaluation metric calculations.
</details>

<details>
<summary>6. Final Bonus: Signal Processing</summary>

*   Analyzed a noisy LED light signal from `light_sig_with_noise.csv`.
*   Identified fluctuations, recurring noise, and intensity decline through visualization (histograms, KDE, box plots, line plots).
*   Cleaned the signal by:
    *   Removing outliers using the IQR method.
    *   Applying a Savitzky-Golay filter for smoothing.
    *   (Experimented with) Applying a low-pass Butterworth filter.
*   Visualized the original, smoothed, and filtered signals to demonstrate the noise reduction process.
</details>

## Technologies Used

*   Python 3
*   Pandas: Data manipulation and analysis.
*   NumPy: Numerical operations.
*   Matplotlib & Seaborn: Data visualization.
*   Scikit-learn: Machine learning (preprocessing, models, metrics, cross-validation, tuning).
*   SciPy: Statistical tests and signal processing filters (Savitzky-Golay, Butterworth).
*   unittest: Python standard library for unit testing.
*   Jupyter Notebook: Project environment.

## How to Use

1.  Clone the repository.
2.  Ensure you have Python and the necessary libraries installed (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy).
3.  Open and run the Jupyter Notebook (`.ipynb` file) in a Jupyter environment (Jupyter Lab, Jupyter Notebook, VS Code with Python extension, etc.).
