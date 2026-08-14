import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read raw IoT device telemetry
telemetry_df = spark.createDataFrame(
    [
        ("DEV-001", "2026-08-13 10:00:00", 72.5, 45.0),
        ("DEV-001", "2026-08-13 10:00:05", 72.8, 45.1),
        ("DEV-002", "2026-08-13 10:00:01", 68.1, 50.2),
    ],
    ["device_id", "event_ts", "temperature", "humidity"]
)

# Cast event_ts from StringType to TimestampType for correct temporal ordering
# (lexicographic ordering on ISO-8601 strings would technically work here, but
# using a native TimestampType is more robust and aligns with the Silver-layer
# requirement to 'convert string timestamps to native time structures'.)
telemetry_df = telemetry_df.withColumn(
    "event_ts",
    F.to_timestamp(F.col("event_ts"), "yyyy-MM-dd HH:mm:ss")
)

# Silver-layer validation: drop records with null sensor readings or
# out-of-range humidity values, per the Device Telemetry Pipeline runbook.
telemetry_df = telemetry_df.filter(
    F.col("temperature").isNotNull() &
    F.col("humidity").isNotNull() &
    (F.col("humidity") >= 0) &
    (F.col("humidity") <= 100)
)

# Define window to group events per device, ordered by event timestamp
# descending so that row_number() == 1 selects the LATEST reading per device.
# FIX: added .orderBy(F.col("event_ts").desc()) — row_number() requires an
# ORDER BY clause; the previous partition-only WindowSpec caused:
#   AnalysisException: Window function row_number() requires window to be ordered
device_window = Window.partitionBy("device_id").orderBy(F.col("event_ts").desc())

# Filter duplicate telemetry readings, keeping the most recent record per device
deduplicated_df = telemetry_df.withColumn(
    "row_num",
    F.row_number().over(device_window)
).filter(F.col("row_num") == 1).drop("row_num")

# Write the deduplicated telemetry to S3 in Parquet format.
# Using a distributed write instead of collect() is the correct AWS Glue
# pattern — collect() pulls all data to the driver and does not scale.
deduplicated_df.write.mode("overwrite").parquet(
    "s3://YOUR-BUCKET/output/telemetry_deduplicated/"
)

job.commit()
