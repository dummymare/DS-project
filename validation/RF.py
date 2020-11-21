from azureml.core import Run
import pandas as pd
import numpy as np
import argparse
import gc

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nLearner', type=int, dest='nLearner', default=100)
args = parser.parse_args()

#Get run context
run = Run.get_context()
dat = run.input_datasets['trainset'].to_pandas_dataframe()
testset = run.input_datasets['testset'].to_pandas_dataframe()
dat = dat.append(testset)

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
dummiesTest = run.input_datasets['dummiesTest'].to_pandas_dataframe().iloc[:, 1:].values
dummies = np.vstack((dummies, dummiesTest))

#Continous variables
dat = dat.loc[:, ['BLOCK', 'LOT', 'ZIP CODE','RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','SALE PRICE', 'IS BUILDING CLASS CHANGED', 'IS TAX CLASS CHANGED']]

y = dat['SALE PRICE'].values
del dat['SALE PRICE']
x = dat.values

x = np.hstack((x, encoded, dummies))
x = MinMaxScaler(feature_range=(min(y), max(y))).fit(x).transform(x)

#Free memory
del encoded
del testset
del dummiesTest
gc.collect()

#Training
Regressor = RandomForestRegressor(max_depth=10, n_estimators=40, n_jobs=4)
test_index = 111512
model = Regressor.fit(x[:test_index, :], y[:test_index])

result = model.predict(x[test_index:, :])
trainResult = model.predict(x[:test_index, :])

#output
run.log("Train R square", r2_score(y[:test_index], result))
run.log("Train Mean squared error", mean_squared_error(y[:test_index], result))
run.log("Train Mean absolute error", mean_absolute_error(y[:test_index], result))

run.log("R square", r2_score(y[test_index:], trainResult))
run.log("Mean squared error", mean_squared_error(y[test_index:], trainResult))
run.log("Mean absolute error", mean_absolute_error(y[test_index:], trainResult))

run.complete()