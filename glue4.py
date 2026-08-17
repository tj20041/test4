import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.errors import AnalysisException

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

# Cast event_ts to TimestampType for correct temporal ordering
telemetry_df = telemetry_df.withColumn(
    "event_ts",
    F.to_timestamp(F.col("event_ts"), "yyyy-MM-dd HH:mm:ss")
)

logger.info(
    f"telemetry_df schema: {telemetry_df.schema} | "
    f"row count before deduplication: {telemetry_df.count()}"
)

# Define window to group events per device, ordered by most-recent event first.
# Secondary sort on temperature provides a fully deterministic tie-break when
# two events for the same device share an identical event_ts value.
device_window = (
    Window
    .partitionBy("device_id")
    .orderBy(F.col("event_ts").desc(), F.col("temperature").desc())
)

try:
    # Filter duplicate telemetry readings — keep the latest reading per device
    deduplicated_df = telemetry_df.withColumn(
        "row_num",
        F.row_number().over(device_window)
    ).filter(F.col("row_num") == 1).drop("row_num")

    logger.info(
        f"row count after deduplication: {deduplicated_df.count()}"
    )

    # Process telemetry dataset
    deduplicated_df.collect()

except AnalysisException as e:
    logger.error(f"Window deduplication failed: {e}")
    job.commit()
    sys.exit(1)

job.commit()
