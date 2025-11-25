#importing all necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#importing the dataset
data = pd.read_csv('house_prices.csv').sample(n=1000, random_state=42)

data = data.dropna(subset=['Price (in rupees)']) #dropping the column

features = [
    'Amount(in rupees)', 'location', 'Carpet Area', 'Status',
    'Transaction', 'Furnishing', 'Bathroom', 'Balcony', 'Car Parking',
    'Ownership', 'Super Area'
]
X = data[features]
y = data['Price (in rupees)']

cols_all_nan = [col for col in X.columns if X[col].isnull().all()]
X = X.drop(columns=cols_all_nan)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# using LinearRegression for speed
model = LinearRegression()

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])
#-------------------------------------------------------------------------------------------------------------------
#divding the data into ratio of 80:20 for training and test
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
print(f'Mean Absolute Error: {mae:.2f}') #It will tell the accuracy of my house prediction model
#--------------------------------------------------------------------------------------------------------------
#for sample output 
sample = pd.DataFrame({
    'location': ['Alkapuri'],
    'Carpet Area': [1200],
    'Bathroom': [2],
    'Balcony': [1],
    'Ownership': ['Freehold']
})
#based on features the house price prediction
sample = sample.reindex(columns=X.columns, fill_value=None)
predicted_price = clf.predict(sample)
print(f"Predicted House Price: ₹{predicted_price[0]:,.0f}") 
