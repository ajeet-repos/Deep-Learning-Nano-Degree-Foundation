import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data from the text file
dataset = pd.read_csv('challenge_dataset.csv', header=None)
x_values = dataset[[0]]
y_values = dataset[[1]]

print('values dataset {}'.format(len(x_values)))
print('labels dataset {}'.format(len(y_values)))
print('shape {}'.format(x_values.shape))


# train model on data
body_regression = linear_model.LinearRegression()
body_regression.fit(x_values, y_values)

# visualizing the results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_regression.predict(x_values))
plt.show()