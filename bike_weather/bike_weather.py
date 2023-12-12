from pyspark.sql import SparkSession
from pyspark.sql.functions import explode,col,split,concat,substring,when,isnull,count
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder.appName('bike_weatheer').getOrCreate()

weather=spark.read.option('multiline','true').json('/home/jss/app/spark/data/json_cycle_finale.json')
#bike = spark.read.option('multiline','true').json('/home/jss/app/spark/data/bike.json')

exploded_df = weather.select(explode("response.body.items.item").alias("item"))

# 중첩된 구조를 각각의 컬럼으로 분리
final_df = exploded_df.select(
    col("item.tm").alias("tm"),
    col("item.avgTa").alias("avgTa"),
    col("item.minTa").alias("minTa"),
    col("item.minTaHrmt").alias("minTaHrmt"),
    col("item.maxTa").alias("maxTa"),
    col("item.maxTaHrmt").alias("maxTaHrmt"),
    col("item.mi10MaxRn").alias("mi10MaxRn"),
    col("item.mi10MaxRnHrmt").alias("mi10MaxRnHrmt"),
    col("item.hr1MaxRn").alias("hr1MaxRn"),
    col("item.hr1MaxRnHrmt").alias("hr1MaxRnHrmt"),
    col("item.sumRnDur").alias("sumRnDur"),
    col("item.sumRn").alias("sumRn"),
    col("item.maxInsWs").alias("maxInsWs"),
    col("item.maxInsWsWd").alias("maxInsWsWd"),
    col("item.maxInsWsHrmt").alias("maxInsWsHrmt"),
    col("item.maxWs").alias("maxWs"),
    col("item.maxWsWd").alias("maxWsWd"),
    col("item.maxWsHrmt").alias("maxWsHrmt"),
    col("item.avgWs").alias("avgWs"),
    col("item.hr24SumRws").alias("hr24SumRws"),            
    col("item.maxWd").alias("maxWd"),
    col("item.avgTd").alias("avgTd"),            
    col("item.minRhm").alias("minRhm"),                
    col("item.minRhmHrmt").alias("minRhmHrmt"),
    col("item.avgRhm").alias("avgRhm"),
    col("item.avgPv").alias('avgPv'),
    col("item.avgPa").alias('avgPa'),
    col("item.maxPs").alias('maxPs'),                
    col("item.maxPsHrmt").alias('maxPsHrmt'),
    col("item.minPs").alias('minPs'),
    col("item.minPsHrmt").alias('minPsHrmt'),
    col("item.avgPs").alias('avgPs'),
    col("item.ssDur").alias('ssDur'),
    col("item.sumSsHr").alias('sumSsHr'),
    col("item.hr1MaxIcsrHrmt").alias('hr1MaxIcsrHrmt'),
    col("item.hr1MaxIcsr").alias('hr1MaxIcsr'),
    col("item.sumGsr").alias('sumGsr'),
    col("item.ddMefs").alias('ddMefs'),
    col("item.ddMefsHrmt").alias('ddMefsHrmt'),
    col("item.ddMes").alias('ddMes'),
    col("item.ddMesHrmt").alias('ddMesHrmt'),
    col("item.sumDpthFhsc").alias('sumDpthFhsc'),
    col("item.avgTca").alias('avgTca'),
    col("item.avgLmac").alias('avgLmac'),
    col("item.avgTs").alias('avgTs'),
    col("item.minTg").alias('minTg'),
    col("item.avgCm5Te").alias('avgCm5Te'),
    col("item.avgCm10Te").alias('avgCm10Te'),
    col("item.avgCm20Te").alias('avgCm20Te'),
    col("item.avgCm30Te").alias('avgCm30Te'),
    col("item.avgM05Te").alias('avgM05Te'),
    col("item.avgM10Te").alias('avgM10Te'),
    col("item.avgM15Te").alias('avgM15Te'),
    col("item.avgM30Te").alias('avgM30Te'),
    col("item.avgM50Te").alias('avgM50Te'),
    col("item.sumLrgEv").alias('sumLrgEv'),
    col("item.sumSmlEv").alias('sumSmlEv'),
    col("item.n99Rn").alias('n99Rn'),
    col("item.sumFogDur").alias('sumFogDur')
                    )

final_df = final_df.withColumn("year", split(final_df["tm"], "-").getItem(0))
final_df = final_df.withColumn("month", split(final_df["tm"], "-").getItem(1))
final_df = final_df.withColumn("day", split(final_df["tm"], "-").getItem(2))


final_df = final_df.withColumn('sumRn', when(final_df['sumRn'] == "", 0.0).otherwise(final_df['sumRn']))
final_df = final_df.withColumn('sumRnDur', when(final_df['sumRnDur'] == "", 0.0).otherwise(final_df['sumRnDur']))
final_df = final_df.withColumn('sumFogDur', when(final_df['sumFogDur'] == "", 0.0).otherwise(final_df['sumFogDur']))
final_df = final_df.withColumn('ddMes', when(final_df['ddMes'] == "", 0.0).otherwise(final_df['ddMes']))
final_df = final_df.withColumn('ddMefs', when(final_df['ddMefs'] == "", 0.0).otherwise(final_df['ddMefs']))


final_df = final_df.withColumn('avg_rhm_hour', substring(final_df['minRhmHrmt'], 1, 2))
final_df = final_df.withColumn('avg_rhm_minute', substring(final_df['minRhmHrmt'], 3, 2))
final_df = final_df.withColumn('min_ps_hour', substring(final_df['minPsHrmt'], 1, 2))
final_df = final_df.withColumn('min_ps_minute', substring(final_df['minPsHrmt'], 3, 2))
final_df = final_df.withColumn('max_ws_hour', substring(final_df['maxWsHrmt'], 1, 2))
final_df = final_df.withColumn('max_ws_minute', substring(final_df['maxWsHrmt'], 3, 2))
final_df = final_df.withColumn('max_ta_hour', substring(final_df['maxTaHrmt'], 1, 2))
final_df = final_df.withColumn('max_ta_minute', substring(final_df['maxTaHrmt'], 3, 2))

final_df = final_df.withColumn('min_ta_hour', substring(final_df['maxWsHrmt'], 1, 2))
final_df = final_df.withColumn('min_ta_minute', substring(final_df['maxWsHrmt'], 3, 2))

final_df = final_df.withColumn('ddmefs_hour', substring(final_df['ddMefsHrmt'], 1, 2))
final_df = final_df.withColumn('ddmefs_minute', substring(final_df['ddMefsHrmt'], 3, 2))


selected_final_df = final_df.select('avgTa','maxTa','sumRnDur','sumRn','maxInsWs','maxInsWsWd','maxWs','maxWsWd','avgWs','maxWd','avgTd','avgRhm','avgPv','ssDur','avgTca','avgLmac','avgTs','avgM50Te','Year','Month','sumFogDur','minTg','avg_rhm_hour','avg_rhm_minute','min_ps_hour','min_ps_minute',	'hr24SumRws','max_ws_hour','max_ws_minute','ddMefs','max_ta_hour','max_ta_minute','min_ta_hour','min_ta_minute')
selected_final_df.show()

convert_to_float = ['avgTa', 'maxTa', 'sumRnDur', 'sumRn', 'maxInsWs', 'maxInsWsWd', 'maxWs', 'maxWsWd', 'avgWs', 'maxWd', 'avgTd', 'avgRhm', 'avgPv', 'ssDur', 'avgTca', 'avgLmac', 'avgTs', 'avgM50Te','sumFogDur','ddMefs'] 
convert_to_int = ['Year', 'Month','minTg', 'avg_rhm_hour', 'avg_rhm_minute','min_ps_hour', 'min_ps_minute','hr24SumRws', 'max_ws_hour','max_ws_minute','max_ta_hour','max_ta_minute','min_ta_hour','min_ta_minute']

for col_name in convert_to_float:
    selected_final_df = selected_final_df.withColumn(col_name, col(col_name).cast('float'))

for col_name in convert_to_int:
    selected_final_df = selected_final_df.withColumn(col_name, col(col_name).cast('int'))
    
non_null_counts = selected_final_df.select([count(when(~isnull(c), c)).alias(c) for c in selected_final_df.columns])

non_null_counts.show()
selected_final_df.filter(col("min_ta_minute").isNull()).show()
selected_final_df.printSchema()



assembler = VectorAssembler(
    inputCols=['maxTa', 'sumRnDur', 'sumRn', 'maxInsWs', 'maxInsWsWd', 'maxWs', 'maxWsWd', 'avgWs', 'maxWd', 'avgTd', 'avgRhm', 'avgPv', 'ssDur', 'avgTca', 'avgLmac', 'avgTs', 'avgM50Te', 'Year', 'Month', 'sumFogDur', 'minTg', 'avg_rhm_hour', 'avg_rhm_minute', 'min_ps_hour', 'min_ps_minute', 'hr24SumRws', 'max_ws_hour', 'max_ws_minute', 'ddMefs', 'max_ta_hour', 'max_ta_minute', 'min_ta_hour', 'min_ta_minute'],
    outputCol="features",
    handleInvalid="skip"  # Change this to "keep" if you want to keep nulls in the dataset
)

selected_final_df = assembler.transform(selected_final_df)
(trainingData, testData) =selected_final_df.randomSplit([0.7, 0.3])

# Continue with training the model
# RandomForestRegressor initialization and training
rf = RandomForestRegressor(featuresCol="features", labelCol="avgTa")
model = rf.fit(trainingData)

# Make predictions on the test set
predictions = model.transform(testData)  # Use your test data here

# Evaluate the model
evaluator = RegressionEvaluator(
    labelCol="avgTa", predictionCol="prediction", metricName="rmse"
)

# Calculate RMSE
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data =", rmse)

#'avgTa','maxTa','sumRnDur','sumRn','maxInsWs','maxInsWsWd','maxWs','maxWsWd','avgWs','maxWd','avgTd','avgRhm','avgPv','ssDur','avgTca','avgLmac','avgTs','avgM50Te','Year','Month,sumFogDur,minTg,avg_rhm_hour,avg_rhm_minute,min_ps_hour,min_ps_minute,	hr24SumRws,max_ws_hour,max_ws_minute,ddmefs,max_ta_hour,max_ta_minute,min_ta_hour,min_ta_minute,ddmefs_hour,ddmefs_minute'


