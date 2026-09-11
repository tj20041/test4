import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import row_number, col
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

df_iot = spark.read.parquet("s3://source-bucket/iot_telemetry/")

window_spec = Window.partitionBy("device_id", "sensor_type")

deduplicated_df = df_iot.withColumn("row_num", row_number().over(window_spec)).filter(col("row_num") == 1)

deduplicated_df.write.mode("overwrite").parquet("s3://output-bucket/iot_deduplicated/")
job.commit()
