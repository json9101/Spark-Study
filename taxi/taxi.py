from pyspark.sql import SparkSession
from pyspark.sql.functions import col,year, month, dayofmonth, hour, minute
import seaborn as sns
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName('taxidata').getOrCreate()

df_1 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2022-06.parquet')
df_2 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2022-07.parquet')
df_3 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2022-08.parquet')
df_4 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2022-09.parquet')
df_5 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2022-10.parquet')
df_6= spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2022-11.parquet')
df_7 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2022-12.parquet')
df_8 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2023-01.parquet')
df_9 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2023-02.parquet')
df_10 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2023-03.parquet')
df_11 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2023-04.parquet')
df_12 = spark.read.parquet('/home/jss/app/spark/data/NYC_Taxi/yellow_tripdata_2023-05.parquet')

df1= df_1.union(df_2)
df2 = df1.union(df_3)
df3 = df2.union(df_4)
df4 = df3.union(df_5)
df5 = df4.union(df_6)
df6 = df5.union(df_7)
df7 = df6.union(df_8)
df8 = df7.union(df_9)
df9 = df8.union(df_10)
df10 = df9.union(df_11)
final_df = df10.union(df_12)
#final_df.show()
# tpep_pickup_datetime 컬럼에서 연도와 월 추출하여 'Year'와 'Month' 컬럼 생성
final_df = final_df.withColumn("Year", year("tpep_pickup_datetime"))
final_df = final_df.withColumn("Month", month("tpep_pickup_datetime"))
# 2022년 6월 이상, 2023년 5월 이하의 데이터 필터링
filtered_df = final_df.filter((col("Year") == 2022) & (col("Month") >= 6) |
                              (col("Year") == 2023) & (col("Month") <= 5))


# 필터링된 데이터에서 연도별, 월별 데이터 수 카운트
result_df = filtered_df.groupBy("Year", "Month").count().orderBy("Year", "Month")


# PySpark DataFrame을 Pandas DataFrame으로 변환
result_pd = result_df.toPandas()

# 연도와 월을 합쳐 새로운 'Period' 컬럼 생성 (선택 사항)
result_pd['Period'] = result_pd['Year'].astype(str) + '-' + result_pd['Month'].astype(str)
plt.switch_backend('TkAgg')

# Seaborn을 사용하여 라인 그래프 생성
plt.figure(figsize=(10, 6))
sns.lineplot(data=result_pd, x='Period', y='count', marker='o')
plt.title('Year-Month_Count')
plt.xlabel('Year-Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()

# null_counts = [final_df.filter(col(c).isNull()).count() for c in final_df.columns]
# for i, col_name in enumerate(final_df.columns):
#      print(f"Column '{col_name}' has {null_counts[i]} null values.")

#final_df.printSchema()
#print(final_df.count() - final_df.na.drop().count())
# # airport_fee 열의 null 값을 필터링
filtered_df = filtered_df.filter(col("Passenger_count").isNotNull())

# # 결과 확인
#filtered_df.show()
filtered= filtered_df.filter(((col("Fare_amount") <= 0) & (col("Payment_type") > 2))|((col("Fare_amount")> 0) & (col("Payment_type")<= 2)))
filtered.show()
# Year와 Month로 그룹화하고 Trip_distance의 평균 계산
distance_result = filtered.groupBy("Year", "Month").agg({"Trip_distance": "mean"}).orderBy("Year", "Month")
distance_result.show()
# PySpark DataFrame을 Pandas DataFrame으로 변환
distance_result_pd = distance_result.toPandas()

# 연도와 월을 합쳐 새로운 'Period' 컬럼 생성 (선택 사항)
distance_result_pd['Period'] = distance_result_pd['Year'].astype(str) + '-' + distance_result_pd['Month'].astype(str)


# Seaborn을 사용하여 라인 그래프 생성
plt.figure(figsize=(10, 6))
sns.lineplot(data=distance_result_pd, x='Period', y='avg(Trip_distance)', marker='o')
plt.title('Year-Month_mile')
plt.xlabel('Year-Month')
plt.ylabel('Mile')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#cleaned_df = filtered_df.filter((filtered_df["Trip_distance"] > 0.0) & (filtered_df["Fare_amount"] != 0.0))
#filtered_df1 = filtered_df.filter(filtered_df["Total_amount"] != 0.0)
#cleaned_df.show()


# 결과 확인
# cleaned_df.show()
# null_counts = [cleaned_df.filter(col(c).isNull()).count() for c in cleaned_df.columns]
# for i, col_name in enumerate(final_df.columns):
#     print(f"Column '{col_name}' has {null_counts[i]} null values.")

# # airport_fee 열의 null 값을 필터링
#f=filtered_df.filter(col("Fare_amount")==0)

# # 결과 확인
#f.show()
spark.stop()