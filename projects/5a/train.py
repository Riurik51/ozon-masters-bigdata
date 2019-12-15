from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
import sys
from model import pipeline, sklearn_est, vectorToArray, predict
import pickle

train_path = sys.argv[1]
pipeline_path = sys.argv[2]
model_path = sys.argv[3]

train = spark.read.json(train_path)
pipeline_model = pipeline.fit(train)
train = pipeline_model.transform(train)
train = train.withColumn("features_array", vectorToArray("features")).localCheckpoint()
df = train.select('label', 'features_array').toPandas()
sklearn_est = sklearn_est.fit(df['features_array'].tolist(), df['label'].tolist())

pipeline_model.write().overwrite().save(pipeline_path)

with open(model_path, "wb") as f:
    pickle.dump(sklearn_est, f)
