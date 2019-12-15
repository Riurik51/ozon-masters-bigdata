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

pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    count_vectorizer
])

sklearn_est = LogisticRegression()

@F.pandas_udf(DoubleType())
def predict(series):
    # Необходимо сделать преобразования, потому что на вход приходит pd.Series(list)
    predictions = sklearn_est.value.predict(np.array(series.tolist()))
    return pd.Series(predictions)

@F.udf(ArrayType(DoubleType()))
def vectorToArray(row):
    return row.toArray().tolist()

