import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.utils import AnalysisException

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
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

    # Cast the event_ts string column to a proper timestamp type so that
    # ordering (and any downstream time-based logic) is deterministic and
    # correct instead of relying on lexicographic string ordering.
    telemetry_df = telemetry_df.withColumn(
        "event_ts",
        F.to_timestamp(F.col("event_ts"), "yyyy-MM-dd HH:mm:ss")
    )

    # Define window to group events per device. row_number() is a ranking
    # function and REQUIRES a deterministic orderBy clause, otherwise
    # Spark's analyzer raises an AnalysisException when the plan is resolved.
    device_window = Window.partitionBy("device_id").orderBy(F.col("event_ts").asc())

    # Filter duplicate telemetry readings, keeping the earliest reading per device
    deduplicated_df = telemetry_df.withColumn(
        "row_num",
        F.row_number().over(device_window)
    ).filter(F.col("row_num") == 1).drop("row_num")

    # Process telemetry dataset
    deduplicated_df.collect()

except AnalysisException as e:
    logger.error(
        "AnalysisException while building/executing deduplicated_df. "
        "This usually indicates a malformed Window spec (e.g. missing orderBy "
        "on a ranking function) or a schema resolution issue. Error: %s", str(e)
    )
    try:
        logger.error("telemetry_df schema: %s", telemetry_df.schema.simpleString())
    except Exception:
        logger.error("telemetry_df was not available for schema logging.")
    sys.exit(1)
except Exception as e:
    logger.error("Unexpected error during telemetry deduplication pipeline: %s", str(e))
    sys.exit(1)

job.commit()
