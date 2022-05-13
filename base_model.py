# Best resources
# https://www.kaggle.com/someadityamandal/bitcoin-time-series-forecasting
# the paper: https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach
# https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo
import timeit

import numpy as np
import pandas as pd
import numba as nb

import mysql.connector as sql
import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.show(block=True)

# adjust pandas output
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

# CUDA params if we're using a GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Open connection to MySQL dataset
host = 'makeitrain-master-instance-1.cqkqu8ldpuqf.us-east-1.rds.amazonaws.com'
conn = sql.connect(host=host, database='master', user='admin', password='UeIHjw7Vq97jef4')
df = pd.read_sql('SELECT * FROM btc_prices_1m ORDER BY date DESC LIMIT 1576800', conn) # 3 years

df_master = df.copy()

# create moving averages
df['sma_180d'] = df.close.rolling(259200).mean() # 180d moving average based on minute
df['diff_close'] = df.close.diff()

# TODO: impute missing values by using previous close as OHLC

def count_upticks(x):
    upticks = 0
    for close in x:
        if close > 0:
            upticks += 1
    return upticks

@nb.njit(nogil=True)
def count_upticks_np(x):
    return np.sum(np.where(x > 0, 1, 0))

@nb.njit(nogil=True)
def pct_chg(x): # this takes in a Pandas Series, of length specified in .rolling(len)
    return x[-1]/x[0] - 1

@nb.njit(nogil=True)
def tot_abs_pct_change(x):
    return np.sum(np.abs(x)) / x[0]
    #return x.abs().sum() / x.iat[0]

df['upticks_last20'] = df.diff_close.rolling(20).apply(count_upticks_np, engine="numba", raw=True) # raw just passes it as a numpy Array instead of a pd.Series
df['upticks_last10'] = df.diff_close.rolling(10).apply(count_upticks_np, engine="numba", raw=True)
df['upticks_last5'] = df.diff_close.rolling(5).apply(count_upticks_np, engine="numba", raw=True)
print('foo')

# df['totchange_last20a'] = df.diff_close.rolling(20).apply(tot_abs_pct_change, engine="numba", raw=True)
# df['totchange_last10'] = df.diff_close.rolling(10).apply(tot_abs_pct_change)
# df['totchange_last5'] = df.diff_close.rolling(5).apply(tot_abs_pct_change)

df['pct_chg_last2'] = df.close.rolling(2).apply(pct_chg, engine="numba", raw=True)
df['pct_chg_last5'] = df.close.rolling(5).apply(pct_chg, engine="numba", raw=True)
df['pct_chg_last10'] = df.close.rolling(10).apply(pct_chg, engine="numba", raw=True)

df['hour'] = df.date.dt.hour
df['month'] = df.date.dt.month

df.drop('date', axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)



@nb.njit(nogil=True)
def increased_xpct(x):
    return np.where(x[-1]/x[0] >= 1.01, 1, 0)

# compute target
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=10)
df['target'] = df.close.rolling(window=indexer, min_periods=10).apply(increased_xpct, engine="numba", raw=True)
# df.target.value_counts(dropna=False)
# df.target.describe()

# TODO: try "total positive", "total negative", "total movement" as a %... in other words how much up and down abs

df_ready = df[['upticks_last5', 'upticks_last10',
               'upticks_last20', 'hour', 'pct_chg_last2', 'pct_chg_last5', 'pct_chg_last10',
               'target']]
df_ready = df_ready.dropna()

# try XGBoost classification
X, y = df_ready.iloc[:,:-1], df_ready.iloc[:,-1] # features and labels

data_dmatrix = xgb.DMatrix(data=X, label=y)

# split into train and test sets – 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# train the model
xgb.set_config(verbosity=2)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# feature importances
print(model.feature_names_in_)
print(model.feature_importances_)
#xgb.plot_importance(model)


# Eval criteria
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
c = confusion_matrix(y_test, y_pred)
a = roc_auc_score(y_test, y_pred)
balanced_accuracy_score(y_test, y_pred)



# plot the trend
plt.plot(df.close)
plt.show()

