from pyspark.sql import SparkSession
import os

def spark_job_example():
    # Spark 세션 생성
    spark = SparkSession.builder.master("local[*]").appName("PythonSparkExample").getOrCreate()

    # 간단한 Spark 작업 예제
    data = [("Alice", 1), ("Bob", 2), ("Catherine", 3)]
    df = spark.createDataFrame(data, ["Name", "Value"])
    df.show()


    # 현재 작업 디렉토리의 절대 경로를 사용하여 디렉토리 생성
    directory = '/save/parquet'
    absolute_directory = os.path.abspath(directory)
    os.makedirs(absolute_directory, exist_ok=True)

    # DataFrame을 Parquet 파일로 저장
    df.to_parquet(os.path.join(absolute_directory, 'test.parquet'))


    # Spark 세션 종료
    spark.stop()

if __name__ == "__main__":
    spark_job_example()
