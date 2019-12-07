from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
import sys
from model import pipeline

train_path = sys.argv[1]
model_path = sys.argv[2]

train = spark.read.json(train_path)
pipeline_model = pipeline.fit(train)

pipeline_model.write().overwrite().save(model_path)