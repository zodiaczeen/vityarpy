# House Price Prediction using Machine Learning

# Overview
This project predicts house prices based on key features of properties using a simple and efficient machine learning model. It is designed for quick setup and demonstrates core regression techniques on a real estate dataset.

# Features
- Predicts house prices from location, area, number of bathrooms, balconies, and ownership type.
- Handles missing data automatically.
- Accepts user input for new house features to estimate price.
- Evaluates model with Mean Absolute Error (MAE) for accuracy.

# Technologies
- Python
- Pandas and NumPy for data manipulation
- scikit-learn for preprocessing and machine learning
- Jupyter Notebook or Python script

# Installation
1. Clone the repository.
2. Install required libraries:
    ```
    pip install pandas numpy scikit-learn
    ```
3. Place your dataset file (e.g., `house_prices.csv`) in the project folder.

# How to Run
1. Edit the main script to list your feature columns and load your dataset.
2. Run the script from command line or in Jupyter:
    ```
    python house_price_predictor.py
    ```
3. The script will output model evaluation (MAE) and predict prices for sample/new properties.

# Example Usage
sample = pd.DataFrame({
'location': ['Alkapuri'],
'Carpet Area': ,
'Bathroom':,​
'Balcony':,​
'Ownership': ['Freehold']
})
sample = sample.reindex(columns=X.columns, fill_value=None)
predicted_price = clf.predict(sample)
print(f"Predicted House Price: ₹{predicted_price:,.0f}")

# Testing
- The script prints Mean Absolute Error to indicate prediction quality.
- You can test with other sample data to check outputs.

# References
- [Scikit-learn regression documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- Public housing datasets(kaggle)

    # Dataset
The full training dataset `house_prices.csv` is not included in this repository due to file size constraints.  
- Please contact the project author or use the shared (https://www.kaggle.com/datasets/juhibhojani/house-price?resource=download) to obtain the dataset.
- Place the file in the root project directory before running code.


