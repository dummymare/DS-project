from azureml.core import Run
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

run = Run.get_context()
dat = run.input_datasets['data'].to_pandas_dataframe()

dat['SALE PRICE'] = np.log(dat['SALE PRICE'])
dat['LAND SQUARE FEET'] = np.log(dat['LAND SQUARE FEET'])
dat['GROSS SQUARE FEET'] = np.log(dat['GROSS SQUARE FEET'])
dat = dat.loc[:, ['BOROUGH', 'BLOCK', 'LOT', 'ZIP CODE','RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','SALE PRICE','BUILDING CLASS SALE QTF', 'IS BUILDING CLASS CHANGED', 'IS TAX CLASS CHANGED']]

y = dat['SALE PRICE']
del dat['SALE PRICE']
x = dat

DTregressor = regr_2 = DecisionTreeRegressor(min_samples_split=10)
model = DTregressor.fit(x, y)
result = model.predict(x)

run.log("R square", r2_score(y, result))
run.log("Mean squared error", mean_squared_error(y, result))
run.log("Mean absolute error", mean_absolute_error(y, result))
run.complete()