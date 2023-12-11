from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max

spark = SparkSession.builder.appName("PythonSpark").getOrCreate()

movie = spark.read.csv("../data/ml-latest/movies.csv", header=True, inferSchema=True)
genome_tag=spark.read.csv("../data/ml-latest/genome-tags.csv", header=True, inferSchema=True)
genome_score=spark.read.csv("../data/ml-latest/genome-scores.csv", header=True, inferSchema=True)

join_data = movie.join(genome_score,on="movieId",how='Inner')
total_data = join_data.join(genome_tag,on='tagId',how="Inner")
max_relevance_by_title = total_data.groupBy('title').agg(max('relevance').alias('max_relevance'))

result = max_relevance_by_title.join(total_data.alias('td'), ['title'], 'inner') \
                               .filter((col('max_relevance') == col('td.relevance'))) \
                               .select(max_relevance_by_title['title'], col('td.tag'),max_relevance_by_title['max_relevance'])

result.show()
print(result.count())

spark.stop()