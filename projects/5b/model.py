from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from sklearn.linear_model import LogisticRegression
import pyspark.sql.functions as F
import pandas as pd

tokenizer = RegexTokenizer(inputCol="reviewText", pattern='[\s\p{Punct}]', outputCol="reviewords")
stop_words = StopWordsRemover.loadDefaultStopWords("english")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="words_filtered", stopWords=stop_words)
count_vectorizer = CountVectorizer(vocabSize=700, minDF=0.001, inputCol=swr.getOutputCol(), outputCol="features", binary=True)

@F.udf(ArrayType(DoubleType()))
def vectorToArray(row):
    return row.toArray().tolist()

class HasSklearnModel(Params):
    sklearn_model = Param(Params._dummy(), "sklearn_model", "sklearn_model",
        typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasSklearnModel, self).__init__()
        self._setDefault(sklearn_model=None)

    def setSklearnModel(self, value):
        return self._set(sklearn_model=value)

    def getSklearnModel(self):
        return self.getOrDefault(self.sklearn_model)

class SklearnEstimatorModel(Model, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasSklearnModel, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, sklearn_model=None, featuresCol="features", labelCol="label", predictionCol="prediction"):
        super(SklearnEstimatorModel, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        dataset = dataset.withColumn("features_array", vectorToArray("features")).localCheckpoint()
        estimator = loads(base64.b64decode(self.getSklearnModel().encode('utf-8')))
        est_broadcast = spark.sparkContext.broadcast(estimator)
        @F.pandas_udf(DoubleType())
        def predict(series):
            predictions = est_broadcast.value.predict(np.array(series.tolist()))
            return pd.Series(predictions)
        return dataset.withColumn(self.getPredictionCol(), predict(self.getFeaturesCol()))


class SklearnEstimator(Estimator, HasFeaturesCol, HasPredictionCol, HasSklearnModel, HasLabelCol, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, featuresCol="features", predictionCol="prediction", labelCol="label"):
        super(SklearnEstimator, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def _fit(self, dataset):
        dataset = dataset.withColumn("features_array", vectorToArray(self.getFeaturesCol()))
        local_dataset = dataset.select("features_array", self.getLabelCol()).toPandas()
        self.est = LogisticRegression()
        self.est.fit(np.array(local_dataset["features_array"].tolist()), local_dataset[self.getLabelCol()])
        model_string = base64.b64encode(dumps(self.est)).decode('utf-8')
        return SklearnEstimatorModel(sklearn_model=model_string, predictionCol=self.getPredictionCol(),
                                         featuresCol='features_array', labelCol=self.getLabelCol())


spark_est = SklearnEstimator(featuresCol=count_vectorizer.getOutputCol())

pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    count_vectorizer,
    spark_est
])

