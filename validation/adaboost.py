from azureml.core import Run
import pandas as pd
import numpy as np
import argparse
import gc

from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--C', type=int, dest='regular', default=100)
parser.add_argument('--epsilon', type=float, dest='eps', default=0.1)
args = parser.parse_args()

#Get run context
run = Run.get_context()
dat = run.input_datasets['data'].to_pandas_dataframe()

#Data transformaton
dat['RESIDENTIAL UNITS'] = np.log(dat['RESIDENTIAL UNITS']+0.1)
dat['COMMERCIAL UNITS'] = np.log(dat['COMMERCIAL UNITS']+0.1)
dat['TOTAL UNITS'] = np.log(dat['TOTAL UNITS']+0.1)

dat['BLOCK'] = np.sqrt(dat['BLOCK'])
dat['LOT'] = np.log(dat['LOT'])

dat['SALE PRICE'] = np.log(dat['SALE PRICE'])
dat['LAND SQUARE FEET'] = np.log(dat['LAND SQUARE FEET'])
dat['GROSS SQUARE FEET'] = np.log(dat['GROSS SQUARE FEET'])

#One hot encoding
cateVars = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'TAX CLASS AT TIME OF SALE']
encoded = OneHotEncoder().fit(dat.loc[:, cateVars]).transform(dat.loc[:, cateVars]).toarray()
dummies = run.input_datasets['dummies'].to_pandas_dataframe().iloc[:, 1:].values

#Continous variables
dat = dat.loc[:, ['BLOCK', 'LOT', 'ZIP CODE','RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','SALE PRICE', 'IS BUILDING CLASS CHANGED', 'IS TAX CLASS CHANGED']]

y = dat['SALE PRICE'].values
del dat['SALE PRICE']
x = dat.values
x = np.hstack((x, encoded, dummies))
x = MinMaxScaler(feature_range=(min(y), max(y))).fit(x).transform(x)

#Free memory
del encoded
gc.collect()

#Training
kf = KFold(n_splits = 5, shuffle = True)
Regressor = LinearSVR(C=args.regular, epsilon=args.eps)

trainR2s = []
trainMSEs = []
trainMAEs = []

R2s = []
MSEs = []
MAEs = []

for train_index, test_index in kf.split(x):
    model = Regressor.fit(x[train_index], y[train_index])

    result = model.predict(x[test_index])
    trainResult = model.predict(x[train_index])

    trainR2s.append(r2_score(y[train_index], trainResult))
    trainMSEs.append(mean_squared_error(y[train_index], trainResult))
    trainMAEs.append(mean_absolute_error(y[train_index], trainResult))

    R2s.append(r2_score(y[test_index], result))
    MSEs.append(mean_squared_error(y[test_index], result))
    MAEs.append(mean_absolute_error(y[test_index], result))

#output
run.log("Train R square", sum(trainR2s)/len(R2s))
run.log("Train Mean squared error", sum(trainMSEs)/len(MSEs))
run.log("Train Mean absolute error", sum(trainMAEs)/len(MAEs))

run.log("R square", sum(R2s)/len(R2s))
run.log("Mean squared error", sum(MSEs)/len(MSEs))
run.log("Mean absolute error", sum(MAEs)/len(MAEs))

run.complete()