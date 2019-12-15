import pandas as pd
from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasPredictionCol
from model import predict, vectorToArray, sklearn_est

class SklearnEstimatorModel(Model, HasFeaturesCol, HasLabelCol, HasPredictionCol):
    model_file = Param(Params._dummy(), "model_path",
                      "path to pickled scikit-learn logistic regression model",
                      typeConverter=TypeConverters.toString)
    @keyword_only
    def __init__(self, model_path=None, featuresCol="features_array", labelCol="label", predictionCol="prediction"):
        super(SklearnEstimatorModel, self).__init__()
        if model_path is None:
            raise ValueError("model_path must be specified!")
        with open(model_path, "rb") as f:
            self.estimator = pickle.load(f)
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def _transform(self, dataset):
        dataset = dataset.withColumn("features_array", vectorToArray("features")).localCheckpoint()
        return dataset.withColumn(self.getPredictionCol(), predict(self.getFeaturesCol()))
