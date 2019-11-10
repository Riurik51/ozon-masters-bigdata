#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')

numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]
fields = ["id"] + numeric_features + categorical_features

mask = [False] * 41
column = [6, 9, 13, 16, 17, 19]
for i in range(len(column)):
    column[i] += 13
for i in column:
    mask[i] = True
for i in range(1,12):
    mask[i] = True

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("2.joblib")


#read and infere
read_opts=dict(
        sep='\t', names=fields, index_col=False, header=None,
        iterator=True, chunksize=200
)

for df in pd.read_csv(sys.stdin, **read_opts):
    pred = model.predict_proba(df.loc[:, mask])[:, 1]
    out = zip(df.id, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
