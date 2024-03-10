from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the housing dataset
from sklearn import datasets
cali = datasets.load_iris()
X = cali.data
y = cali.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor on the training set
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Calculate the mean squared error of the regressor
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
