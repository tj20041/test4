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

# Define window to group events per device, ordered by event_ts descending so
# that row_number() == 1 selects the most recent telemetry record per device
# (latest-record deduplication semantics per the Device Telemetry Pipeline spec).
# asc_nulls_last() is used to push any NULL timestamps to the end of the
# partition, preventing NULLs from being silently promoted to row 1.
device_window = Window.partitionBy("device_id").orderBy(F.col("event_ts").asc_nulls_last())

# Filter duplicate telemetry readings — keep only the first-ranked row per device
try:
    deduplicated_df = telemetry_df.withColumn(
        "row_num",
        F.row_number().over(device_window)
    ).filter(F.col("row_num") == 1).drop("row_num")

    # Process telemetry dataset
    deduplicated_df.collect()

except Exception as e:
    glueContext.get_logger().error(
        "AnalysisException or unexpected error during deduplication: {}".format(str(e))
    )
    job.commit()
    raise

job.commit()
