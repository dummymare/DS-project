from azureml.core import Run
import pandas as pd
import numpy as np
import argparse
import gc

from sklearn.svm import LinearSVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

run = Run.get_context()
dat = run.input_datasets['data'].to_pandas_dataframe()
dummies = run.input_datasets['dummies'].to_pandas_dataframe().iloc[1:, 1:].values
dummies[(dummies == 'True')] = True
dummies[(dummies == 'False')] = False

#Data transformaton
dat['RESIDENTIAL UNITS'] = np.log(dat['RESIDENTIAL UNITS']+0.1)
dat['COMMERCIAL UNITS'] = np.log(dat['COMMERCIAL UNITS']+0.1)
dat['TOTAL UNITS'] = np.log(dat['TOTAL UNITS']+0.1)

dat['BLOCK'] = np.sqrt(dat['BLOCK'])
dat['LOT'] = np.sqrt(dat['LOT'])

dat['SALE PRICE'] = np.log(dat['SALE PRICE'])
dat['LAND SQUARE FEET'] = np.log(dat['LAND SQUARE FEET'])
dat['GROSS SQUARE FEET'] = np.log(dat['GROSS SQUARE FEET'])

#One hot encoding
cateVars = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'TAX CLASS AT TIME OF SALE']
encoded = OneHotEncoder().fit(dat.loc[:, cateVars]).transform(dat.loc[:, cateVars]).toarray()

#Continous variables
dat = dat.loc[:, ['BLOCK', 'LOT', 'ZIP CODE','RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','SALE PRICE', 'IS BUILDING CLASS CHANGED', 'IS TAX CLASS CHANGED']]

y = dat['SALE PRICE'].values
del dat['SALE PRICE']
x = dat.values
x = np.hstack((x, encoded, dummies))

#Free memory
del encoded
del dummies
gc.collect()

x = MinMaxScaler(feature_range=(min(y), max(y))).fit(x).transform(x)

Regressor = LinearSVR(C=100, epsilon=.1)
model = Regressor.fit(x, y)
result = model.predict(x)

run.log("R square", r2_score(y, result))
run.log("Mean squared error", mean_squared_error(y, result))
run.log("Mean absolute error", mean_absolute_error(y, result))
run.complete()