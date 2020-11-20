from azureml.core import Run
import pandas as pd
import numpy as np

from sklearn.feature_selection import f_regression
from sklearn.preprocessing import OneHotEncoder

run = Run.get_context()
dat = run.input_datasets['data'].to_pandas_dataframe()

dat = dat.loc[:, ['SALE PRICE', 'STREET NAME']]

y = np.log(dat['SALE PRICE'].values)
del dat['SALE PRICE']
x = OneHotEncoder().fit(dat['STREET NAME'].values.reshape(-1,1)).transform(dat['STREET NAME'].values.reshape(-1,1)).toarray()

F, pval = f_regression(x, y)

#output
run.log("Array shape", pval.shape)
run.log("Significant items", pval[(pval<0.05)].shape)
run.complete()