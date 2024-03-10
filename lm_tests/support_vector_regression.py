# Importing necessary libraries
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generating sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel()

# Adding noise to targets
y[::5] += 3 * (0.5 - np.random.rand(20))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()  # Reshape for compatibility

# Creating SVR model
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)  # You can adjust hyperparameters as needed

# Training the SVR model
svr.fit(X_train, y_train)

# Predicting on the test set
y_pred = svr.predict(sc_X.transform(X_test))

# Rescaling the predictions
y_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1))

# Plotting
plt.scatter(X, y, color='black', label='Data')
plt.plot(X_test, y_pred, color='red', label='SVR RBF')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
