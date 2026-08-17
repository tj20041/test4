import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
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

# Cast event_ts from StringType to TimestampType for correct temporal ordering.
# Lexicographic string ordering works for ISO-8601 strings but is fragile;
# an explicit cast ensures correctness if the upstream format ever changes.
telemetry_df = telemetry_df.withColumn(
    "event_ts",
    F.to_timestamp(F.col("event_ts"), "yyyy-MM-dd HH:mm:ss")
)

# Define window to group events per device, ordered by event_ts descending
# so that row_num = 1 corresponds to the most recent telemetry record per device.
# row_number() requires an ORDER BY clause; omitting it raises AnalysisException.
device_window = Window.partitionBy("device_id").orderBy(F.col("event_ts").desc())

# Filter duplicate telemetry readings, keeping only the latest event per device
deduplicated_df = telemetry_df.withColumn(
    "row_num",
    F.row_number().over(device_window)
).filter(F.col("row_num") == 1).drop("row_num")

# Process telemetry dataset
deduplicated_df.collect()

job.commit()
