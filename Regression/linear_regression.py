import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data from the text file
dataset = pd.read_fwf('brain_body.txt')
x_values = dataset[['Brain']]
y_values = dataset[['Body']]

# train model on data
body_regression = linear_model.LinearRegression()
body_regression.fit(x_values, y_values)

# visualizing the results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_regression.predict(x_values))
plt.show()