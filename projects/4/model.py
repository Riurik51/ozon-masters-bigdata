from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml import Pipeline

tokenizer = RegexTokenizer(inputCol="reviewText", pattern='[\s\p{Punct}]', outputCol="reviewords")
stop_words = StopWordsRemover.loadDefaultStopWords("english")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="words_filtered", stopWords=stop_words)
count_vectorizer = CountVectorizer(vocabSize=7000, minDF=0.0001, inputCol=swr.getOutputCol(), outputCol="word_vector", binary=True)
lr = LinearRegression(featuresCol=count_vectorizer.getOutputCol(), labelCol='overall', maxIter=10)

pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    count_vectorizer,
    lr
])