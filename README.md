# Customer Churn Prediction

## Project Overview

This project aims to predict customer churn for a telecom company using machine learning. It involves data preprocessing, feature engineering, model training, and evaluation. The project utilizes the ExtraTreesClassifier model for predictions.

## Dataset

The project uses the "dataset.csv" file, which contains customer information and churn status. The dataset is assumed to be located in the same directory as the notebook.

## Requirements

To run this project, you need the following libraries:
pandas
numpy 
scikit-learn 
matplotlib 
seaborn
xgboost
catboost 
imblearn
You can install them using `pip install -r requirements.txt`, where `requirements.txt` contains the above list.

## Usage

1. ## Data Preprocessing

This step involves preparing the raw data for use in model training. The following actions are performed:

 1. **Data Cleaning:**
   - Handling missing values by filling them with appropriate strategies (e.g., imputation using the mean or median).
   - Removing irrelevant or redundant columns if present.

 2. **Feature Engineering:**
   - Transforming categorical features into numerical representations using Label Encoding and One-Hot Encoding.
   - Creating new features from existing ones to potentially improve model performance (e.g., interaction terms).

 3. **Data Scaling:**
   - Applying StandardScaler to standardize numerical features, ensuring they have zero mean and unit variance. This helps improve model stability and performance.

**Specific steps:**

- **Label Encoding:** Used for binary categorical features like 'gender', 'Partner', etc.
- **One-Hot Encoding:** Used for multi-category features like 'InternetService', 'PaymentMethod', etc. This avoids imposing an ordinal relationship on the categories.
- **StandardScaler:** Applied to numerical features like 'tenure', 'MonthlyCharges', etc., to ensure they are on a similar scale.

This preprocessing ensures that the data is in a suitable format for training the machine learning model.

2. ## Model Training

This stage involves training the chosen machine learning model on the preprocessed data.

**Model Selection:**

- The **ExtraTreesClassifier** model has been selected for this project. This is an ensemble learning method that builds multiple decision trees and combines their predictions for improved accuracy.

**Training Process:**

1. **Data Splitting:** The preprocessed data is divided into training and testing sets using `train_test_split`. This allows us to evaluate the model's performance on unseen data.
2. **Model Fitting:** The ExtraTreesClassifier model is trained on the training data using the `fit` method. This process involves adjusting the model's parameters to learn patterns from the data.
3. **Model Saving:** The trained model is saved as "Customer_Churn_Prediction.pkl" using `pickle`. This allows us to reuse the model later without retraining.

**Hyperparameter Tuning:**

- While not explicitly mentioned in the initial code, consider using techniques like GridSearchCV or RandomizedSearchCV to find optimal hyperparameter values for the model. This can further improve its performance.

3. **Prediction:**
   - To make predictions on new data, load the saved model and preprocess the input data in the same way as during training.
   - Use the `predict` method of the loaded model to obtain churn predictions.

## Deployment

The project can be deployed using various methods, such as:

- **Flask:** A web app can be created using Flask to expose the model as an API.
- **Streamlit:** An interactive web app can be built using Streamlit for easier user interaction.
- **Google Cloud Functions:** For serverless deployment, the model can be deployed as a Cloud Function.

Refer to the code and comments for specific deployment instructions.

## Evaluation

The model's performance is evaluated using metrics like accuracy, precision, recall, F1-score, and AUC. Refer to the evaluation section in the notebook for detailed results.

## Contributing

Contributions are welcome! Please feel free to open issues or pull requests to suggest improvements or add new features.

## License

This project is licensed under the MIT License.
