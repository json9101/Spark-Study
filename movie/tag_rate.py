from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import col,avg,dense_rank
from pyspark.sql.functions import corr
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("MovieTags").getOrCreate()

movie = spark.read.csv("../data/ml-latest/movies.csv", header=True, inferSchema=True)
tag=spark.read.csv("../data/ml-latest/tags.csv", header=True, inferSchema=True)
rating=spark.read.csv("../data/ml-latest/ratings.csv", header=True, inferSchema=True)

join_data = movie.join(tag,on="movieId",how='Inner')
total_data = join_data.join(rating,on=['movieId','userId'],how="Inner")

total_data.show()
print(total_data.count())

mean_by_user = total_data.groupBy(['userId', 'tag']).agg(avg('rating').alias('mean_rating'))
windowSpec = Window.partitionBy('userId').orderBy(mean_by_user['mean_rating'].desc())
mean_by_user_ranked = mean_by_user.withColumn('rank', dense_rank().over(windowSpec))
mean_by_user_ranked.show()

top_tags_by_user = mean_by_user_ranked.filter(col('rank') == 1).select('userId', 'tag', 'mean_rating')
top_tags_by_user.show()

# 최고 순위 확인
max_rank = mean_by_user_ranked.groupBy().max('rank').collect()[0][0]

# 최고 순위인 데이터 가져오기
top_ranked_users = mean_by_user_ranked.filter(mean_by_user_ranked['rank'] == max_rank)
top_ranked_users.show()

mean_by_tag = total_data.groupBy('tag').agg(avg('rating').alias('mean_rating'))
windowSpec = Window.orderBy(mean_by_tag['mean_rating'].desc())
mean_by_tag_ranked = mean_by_tag.withColumn('rank', dense_rank().over(windowSpec))
mean_by_tag_ranked.show()

# mean_by_tag와 그룹화된 rank의 최대값을 얻습니다.
tag_max_rank = mean_by_tag_ranked.groupBy().max('rank').collect()[0][0]

# 최하위 순위인 데이터를 얻습니다.
bottom_ranked_tag = mean_by_tag_ranked.filter(mean_by_tag_ranked['rank'] == tag_max_rank)
bottom_ranked_tag.show()

# 'tag' 컬럼을 수치형으로 변환
indexer = StringIndexer(inputCol='tag', outputCol='tag_index')
indexed = indexer.fit(total_data).transform(total_data)

first_value_udf = udf(lambda x: float(x[0]), DoubleType())

encoder = OneHotEncoder(inputCol='tag_index', outputCol='tag_encoded')

result = encoder.fit(indexed).transform(indexed) \
                 .withColumn('tag_encoded_value', first_value_udf('tag_encoded')) \
                 .withColumn('rating_double', F.col('rating').cast('double')) \
                 .select(F.corr('tag_encoded_value', 'rating_double')).collect()[0][0]

print(result)

spark.stop()