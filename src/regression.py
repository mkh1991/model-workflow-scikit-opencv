import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import os
import json
from sklearn import datasets, linear_model
from scikit_checkpoint import ScikitCheckpoint

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

with open(os.environ['INPUT_DIR']+'/config.json') as f:
     config = json.load(f)

# Split the data into training/testing sets
split_val = config['split_val']
diabetes_X_train = diabetes_X[:-1*split_val]
diabetes_X_test = diabetes_X[-1*split_val:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-1*split_val]
diabetes_y_test = diabetes.target[-1*split_val:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
coefficients = regr.coef_[0]
mse = np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)
variance_score = regr.score(diabetes_X_test, diabetes_y_test)
print('Coefficients: \n', coefficients)
# The mean squared error
print("Mean squared error: %.2f"% mse)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % variance_score)

# save the classifier
stats = {"mse": mse,"variance_score":variance_score}
checkpoint = ScikitCheckpoint(os.environ['SNAPSHOTS_DIR'], )
checkpoint.save_model(regr, stats)

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.savefig(os.environ['SHARED_OUTPUT_DIR']+'/performance.png')
