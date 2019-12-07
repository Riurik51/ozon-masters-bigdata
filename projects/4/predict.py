from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

import sys
from pyspark.ml import Pipeline, PipelineModel

predict_path = sys.argv[3]
test_path = sys.argv[2]
model_path = sys.argv[1]

model = PipelineModel.load(model_path)
test = spark.read.json(test_path)

predict = model.transform(test)
predict.write.mode("overwrite").text(predict_path)