import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import TimestampType

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

try:
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
    # (aligns with Silver-layer contract: converting string timestamps to native time structures)
    telemetry_df = telemetry_df.withColumn("event_ts", F.col("event_ts").cast("timestamp"))

    # Define window to group events per device, ordered by event_ts descending
    # so that row_num == 1 selects the most recent reading per device.
    # FIX: added .orderBy(F.col("event_ts").desc()) — row_number() requires an ORDER BY clause.
    device_window = Window.partitionBy("device_id").orderBy(F.col("event_ts").desc())

    # Filter duplicate telemetry readings — keep only the most recent record per device
    deduplicated_df = telemetry_df.withColumn(
        "row_num",
        F.row_number().over(device_window)
    ).filter(F.col("row_num") == 1).drop("row_num")

    # Post-deduplication data-quality assertion:
    # Verify that no device_id appears more than once after deduplication.
    duplicate_count = deduplicated_df.groupBy("device_id").count().filter(F.col("count") > 1).count()
    if duplicate_count > 0:
        logger.warn(
            f"Data quality warning: {duplicate_count} device_id(s) still have more than one row "
            "after deduplication. Check partition key and ordering logic."
        )

    # Process telemetry dataset
    deduplicated_df.collect()

    job.commit()

except Exception as e:
    logger.error(f"Glue job failed with error: {str(e)}")
    raise
