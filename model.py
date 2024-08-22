# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('C:\\Users\\ptush\Downloads\Breast Cancer Wisconsin (Diagnostic) Data Set.csv')

dataset['target'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
X = dataset.drop(columns=['id','diagnosis','target'])
y = dataset['target']

# Drop columns with all NaN values
X = X.dropna(how='all', axis=1)
scaler = StandardScaler()
X = np.array(scaler.fit_transform(X), dtype=np.float64)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LogisticRegression
# Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=5000, random_state=42)
log_reg_model.fit(X, y)


# Saving model to disk
pickle.dump(log_reg_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
##print(model.predict([['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
##       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
##       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
##     'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
##       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
##       'fractal_dimension_se', 'radius_worst', 'texture_worst',
##       'perimeter_worst', 'area_worst', 'smoothness_worst',
##       'compactness_worst', 'concavity_worst', 'concave points_worst',
##       'symmetry_worst', 'fractal_dimension_worst', 'target']]))

print(model.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]))
