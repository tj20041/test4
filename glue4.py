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

# Cast event_ts to TimestampType to ensure correct chronological ordering
# rather than lexicographic string ordering
telemetry_df = telemetry_df.withColumn("event_ts", F.col("event_ts").cast("timestamp"))

# Define window to group events per device, ordered by event_ts descending
# so that row_number() == 1 selects the most recent telemetry record per device
device_window = Window.partitionBy("device_id").orderBy(F.col("event_ts").desc())

# Filter duplicate telemetry readings, keeping only the most recent per device
try:
    deduplicated_df = telemetry_df.withColumn(
        "row_num",
        F.row_number().over(device_window)
    ).filter(F.col("row_num") == 1).drop("row_num")

    # Process telemetry dataset
    deduplicated_df.collect()

except AnalysisException as e:
    logger.error("AnalysisException during deduplication window operation: " + str(e))
    raise

job.commit()
