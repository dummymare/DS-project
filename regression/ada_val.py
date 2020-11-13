from azureml.core import Run
import pandas as pd
import numpy as np
import argparse

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, dest='nTrees', default=100)
args = parser.parse_args()

#Get run context
run = Run.get_context()
dat = run.input_datasets['data'].to_pandas_dataframe()

#Feature processing
dat['SALE PRICE'] = np.log(dat['SALE PRICE'])
dat['LAND SQUARE FEET'] = np.log(dat['LAND SQUARE FEET'])
dat['GROSS SQUARE FEET'] = np.log(dat['GROSS SQUARE FEET'])
dat = dat.loc[:, ['BOROUGH', 'BLOCK', 'LOT', 'ZIP CODE','RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','SALE PRICE','BUILDING CLASS SALE QTF', 'IS BUILDING CLASS CHANGED', 'IS TAX CLASS CHANGED']]

y = dat['SALE PRICE'].values
del dat['SALE PRICE']
x = dat.values

#Training
kf = KFold(n_splits = 5, shuffle = True)
R2s = []
MSEs = []
MAEs = []

for train_index, test_index in kf.split(x):
    regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10), n_estimators=args.nTrees)
    model = regressor.fit(x[train_index], y[train_index])
    result = model.predict(x[test_index])

    R2s.append(r2_score(y[test_index], result))
    MSEs.append(mean_squared_error(y[test_index], result))
    MAEs.append(mean_absolute_error(y[test_index], result))

#output
run.log("R square", sum(R2s)/len(R2s))
run.log("Mean squared error", sum(MSEs)/len(MSEs))
run.log("Mean absolute error", sum(MAEs)/len(MAEs))
run.complete()