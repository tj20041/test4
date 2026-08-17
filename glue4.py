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

# Cast event_ts string to TimestampType to guarantee correct temporal ordering
telemetry_df = telemetry_df.withColumn("event_ts", F.to_timestamp(F.col("event_ts")))

# Define window to group events per device, ordered by most recent event first
# ORDER BY is required by row_number(); descending order retains the latest record per device
device_window = Window.partitionBy("device_id").orderBy(F.col("event_ts").desc())

# Filter duplicate telemetry readings
try:
    deduplicated_df = telemetry_df.withColumn(
        "row_num",
        F.row_number().over(device_window)
    ).filter(F.col("row_num") == 1).drop("row_num")

    # Process telemetry dataset
    deduplicated_df.collect()
except Exception as e:
    glueContext.get_logger().error(
        "Failed during telemetry deduplication window operation: {}".format(str(e))
    )
    job.commit()
    sys.exit(1)

job.commit()
