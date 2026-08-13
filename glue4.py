import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, row_number
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window

# ==========================================
# 0. GLUE INITIALIZATION & LOGGING SETUP
# ==========================================
# Fetch job name passed by the AWS Glue execution environment
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger = logging.getLogger("DeviceTelemetryETL")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if re-run
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.info("Initializing Device Telemetry ETL Job...")

try:
    # ==========================================
    # 1. SCHEMA DEFINITIONS
    # ==========================================
    logger.info("Defining explicit schemas...")
    telemetry_schema = StructType([
        StructField("record_id", IntegerType(), True),
        StructField("device_id", StringType(), True),
        StructField("sensor_type", StringType(), True),
        StructField("reading_value", DoubleType(), True),
        StructField("battery_level", IntegerType(), True),
        StructField("event_timestamp", StringType(), True)
    ])

    # ==========================================
    # 2. BRONZE LAYER (Extraction)
    # ==========================================
    logger.info("Extracting telemetry data into Bronze layer...")
    telemetry_bronze = spark.createDataFrame([
        (1, "DEV-A100", "TEMPERATURE", 72.5, 95, "2024-08-10 10:00:00"),
        (2, "DEV-A100", "TEMPERATURE", 74.1, 94, "2024-08-10 10:15:00"),
        (3, "DEV-B200", "PRESSURE", 101.3, 80, "2024-08-10 10:05:00"),
        (4, "DEV-B200", "PRESSURE", 102.0, 79, "2024-08-10 10:20:00"),
        (5, "DEV-A100", "TEMPERATURE", 73.0, 93, "2024-08-10 10:30:00")
    ], schema=telemetry_schema)

    # ==========================================
    # 3. SILVER LAYER (Transformation & Cleansing)
    # ==========================================
    logger.info("Parsing timestamps and filtering valid sensor readings for Silver layer...")
    telemetry_silver = telemetry_bronze.withColumn(
        "event_ts",
        col("event_timestamp").cast(TimestampType())
    ).filter("battery_level > 0 AND reading_value IS NOT NULL")

    # ==========================================
    # 4. GOLD LAYER (Deduplication / Latest State Reconciliation)
    # ==========================================
    logger.info("Computing latest device state per sensor for Gold layer...")
    window_spec = Window.partitionBy("device_id", "sensor_type")

    df_gold = telemetry_silver.withColumn(
        "row_num",
        row_number().over(window_spec)
    ).filter(
        col("row_num") == 1
    ).drop("row_num").select(
        "device_id",
        "sensor_type",
        "reading_value",
        "battery_level",
        "event_ts"
    )

    logger.info("Pipeline completed successfully.")
    df_gold.show(truncate=False)

    # Commit Glue Job
    job.commit()

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
