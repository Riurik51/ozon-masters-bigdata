from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f
import sys

def shortest_path(v_from, v_to, df, max_path_length=10):
    Curr_queue = graph.where(f'follower == {v_from}')
    Curr_queue = Curr_queue.withColumn('path',
                                       f.concat(f.col('follower'),
                                                f.lit(','),
                                                f.col('id'))
                                      ).drop('follower')
    answer = Curr_queue.where(f'id == {v_to}').select('path')
    for i in range(max_path_length - 1):
        if answer.count() > 0:
            return answer
        Curr_queue = graph.join(Curr_queue.select(
            Curr_queue.id.alias('follower'),
            Curr_queue.path),
                          on='follower',
                          how='right').dropna().drop('follower')
        Curr_queue = Curr_queue.withColumn('path',
                                           f.concat(f.col('path'),
                                                    f.lit(','),
                                                    f.col('id'))
                                          )
        answer = answer.union(Curr_queue.where(f'id == {v_to}').select('path'))
    return answer

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

schema = StructType(fields=[
    StructField("id", IntegerType()),
    StructField("follower", IntegerType())
])

input_file = sys.argv[3]
output_file = sys.argv[4]
v_to = int(sys.argv[2])
v_from = int(sys.argv[1])
graph = spark.read.csv(input_file, sep="\t", schema=schema).cache()
df = shortest_path(v_from, v_to, graph)
df.select("path").write.mode("overwrite").text(output_file)
