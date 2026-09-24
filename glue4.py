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

# Define window to group events per device.
# row_number() is an order-sensitive ranking function and REQUIRES a
# deterministic orderBy() clause on the Window spec, otherwise Spark raises
# an AnalysisException at DAG execution time (GEN-UNCLASSIFIED-ERROR).
# We order by event_ts ascending so the earliest reading per device gets
# row_num == 1, with temperature as a tiebreaker to guarantee full
# determinism when two events share the exact same timestamp.
device_window = Window.partitionBy("device_id").orderBy(
    F.col("event_ts").asc(),
    F.col("temperature").asc()
)

# Filter duplicate telemetry readings, keeping the earliest reading per device
deduplicated_df = telemetry_df.withColumn(
    "row_num",
    F.row_number().over(device_window)
).filter(F.col("row_num") == 1).drop("row_num")

# Persist the deduplicated telemetry dataset to S3 instead of collect().
# collect() pulls all rows to the driver and is unsafe for real IoT
# telemetry volumes in production Glue jobs (risk of driver OOM).
deduplicated_df.write.mode("overwrite").parquet(
    "s3://your-bucket/output/deduplicated_telemetry/"
)

job.commit()
