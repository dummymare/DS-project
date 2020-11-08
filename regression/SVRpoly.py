from azureml.core import Run
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

run = Run.get_context()
dat = run.input_datasets['data'].to_pandas_dataframe()

dat['SALE PRICE'] = np.log(dat['SALE PRICE'])
dat['LAND SQUARE FEET'] = np.log(dat['LAND SQUARE FEET'])
dat['GROSS SQUARE FEET'] = np.log(dat['GROSS SQUARE FEET'])
dat = dat.loc[:, ['ZIP CODE','RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','SALE PRICE','BUILDING CLASS SALE QTF','TAX CODE']]

y = dat['SALE PRICE'].values
del dat['SALE PRICE']
x = dat.values

svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
model = svr_poly.fit(x, y)
result = model.predict(x)

run.log("R square", r2_score(y, result))
run.log("Mean squared error", mean_squared_error(y, result))
run.log("Mean absolute error", mean_absolute_error(y, result))
run.complete()