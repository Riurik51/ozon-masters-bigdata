from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import PipelineModel
from sklearn_wrapper import SklearnEstimatorModel
import sys

model_path = sys.argv[1]
sklearn_model_path = sys.argv[2]
test_path = sys.argv[3]
predict_path = sys.argv[4]

pipeline_model = PipelineModel.load(model_path)

df_test = spark.read.json(test_path)

df_test_transformed = pipeline_model.transform(df_test)

spark_est = SklearnEstimatorModel(model_path=sklearn_model_path)

pred = spark_est.transform(df_test_transformed)
pred.write.mode("overwrite").text(predict_path)
