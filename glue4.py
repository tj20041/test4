import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Explicit schema definition for schema stability across data source changes
telemetry_schema = StructType([
    StructField("device_id", StringType(), False),
    StructField("event_ts", StringType(), True),
    StructField("temperature", DoubleType(), True),
    StructField("humidity", DoubleType(), True)
])

# Read raw IoT device telemetry
telemetry_df = spark.createDataFrame(
    [
        ("DEV-001", "2026-08-13 10:00:00", 72.5, 45.0),
        ("DEV-001", "2026-08-13 10:00:05", 72.8, 45.1),
        ("DEV-002", "2026-08-13 10:00:01", 68.1, 50.2),
    ],
    schema=telemetry_schema
)

# Cast event_ts from StringType to TimestampType for correct temporal ordering
# and downstream compatibility with Glue Data Catalog / Athena range queries
telemetry_df = telemetry_df.withColumn(
    "event_ts",
    F.to_timestamp(F.col("event_ts"), "yyyy-MM-dd HH:mm:ss")
)

# Define window to group events per device, ordered by event_ts descending
# so that row_number() == 1 resolves to the most recent telemetry record per device
# (required by Spark Catalyst: row_number() mandates an explicit ORDER BY clause)
device_window = Window.partitionBy("device_id").orderBy(F.col("event_ts").desc())

# Filter duplicate telemetry readings, retaining only the latest record per device
deduplicated_df = telemetry_df.withColumn(
    "row_num",
    F.row_number().over(device_window)
).filter(F.col("row_num") == 1).drop("row_num")

# Process telemetry dataset
deduplicated_df.collect()

job.commit()
