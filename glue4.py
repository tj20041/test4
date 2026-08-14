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
logger = glueContext.get_logger()
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

# Cast event_ts from string to TimestampType for correct chronological ordering
# (avoids lexicographic string comparison and aligns with Silver-layer conventions)
telemetry_df = telemetry_df.withColumn(
    "event_ts",
    F.to_timestamp(F.col("event_ts"), "yyyy-MM-dd HH:mm:ss")
)

# Validate schema assumptions before deduplication
assert "event_ts" in telemetry_df.columns, "Required column 'event_ts' is missing from telemetry DataFrame"
assert telemetry_df.filter(F.col("event_ts").isNull()).count() == 0, "Null values detected in 'event_ts' column — ordering will be unreliable"

# Define window to group events per device, ordered by most-recent event first.
# row_number() requires an explicit ORDER BY clause; descending event_ts ensures
# row_num = 1 is the latest telemetry reading per device.
# Secondary sort on temperature provides a deterministic tie-breaker for
# events sharing an identical timestamp.
device_window = Window.partitionBy("device_id").orderBy(
    F.col("event_ts").desc(),
    F.col("temperature").desc()
)

try:
    # Filter duplicate telemetry readings — keep only the most recent per device
    deduplicated_df = telemetry_df.withColumn(
        "row_num",
        F.row_number().over(device_window)
    ).filter(F.col("row_num") == 1).drop("row_num")

    # Process telemetry dataset
    deduplicated_df.collect()

except Exception as e:
    logger.error(
        f"Deduplication step failed with error: {str(e)}. "
        "Check that the Window specification includes a valid ORDER BY column "
        "and that 'event_ts' is present and non-null in the source data."
    )
    job.commit()
    raise

job.commit()
