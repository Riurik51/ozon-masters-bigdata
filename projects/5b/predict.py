from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import PipelineModel
import sys

model_path = sys.argv[1]
test_path = sys.argv[2]
predict_path = sys.argv[3]

pipeline_model = PipelineModel.load(model_path)

df_test = spark.read.json(test_path)

pred = pipeline_model.transform(df_test)

pred.write.mode("overwrite").text(predict_path)
