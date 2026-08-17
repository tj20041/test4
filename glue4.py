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

# Cast event_ts from string to TimestampType for correct temporal ordering
telemetry_df = telemetry_df.withColumn("event_ts", F.to_timestamp(F.col("event_ts")))

# Define window to group events per device, ordered by event_ts descending
# so that row_number() == 1 selects the MOST RECENT telemetry record per device
# (Gold-layer latest-state monitoring intent per Device Telemetry Pipeline spec)
device_window = Window.partitionBy("device_id").orderBy(F.col("event_ts").desc())

try:
    # Filter duplicate telemetry readings, keeping the most recent record per device
    deduplicated_df = telemetry_df.withColumn(
        "row_num",
        F.row_number().over(device_window)
    ).filter(F.col("row_num") == 1).drop("row_num")

    # Process telemetry dataset
    results = deduplicated_df.collect()
    logger.info(
        f"[glue4] Deduplication complete. "
        f"Output row count: {len(results)}"
    )
except Exception as e:
    logger.error(
        f"[glue4] Deduplication step failed. "
        f"Error: {str(e)}"
    )
    raise RuntimeError(
        "Telemetry deduplication failed — see preceding error log for details."
    ) from e

job.commit()
