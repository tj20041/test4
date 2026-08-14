import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define schema for input invoices
schema = StructType([
    StructField("invoice_id", StringType(), True),
    StructField("item_amounts", ArrayType(DoubleType()), True),
])

data = [
    ("INV-9001", [120.50, 45.00, 19.99]),
    ("INV-9002", [500.00, 150.25]),
]

invoices_df = spark.createDataFrame(data, schema)

# Calculate total invoice amount across line items.
# F.sum() is a grouped aggregation function and cannot be used inside withColumn()
# to sum the elements of an ArrayType column. The correct PySpark API for
# element-wise array reduction is F.aggregate() (supported in Glue 3.0+ / Spark 3.1+).
# A null/empty-array guard is applied via F.when() to prevent NullPointerException
# or unexpected zero-sum results on sparse invoice records.
processed_invoices = invoices_df.withColumn(
    "total_invoice_amount",
    F.when(
        F.col("item_amounts").isNotNull() & (F.size(F.col("item_amounts")) > 0),
        F.aggregate(
            F.col("item_amounts"),
            F.lit(0.0).cast("double"),
            lambda acc, x: acc + x
        )
    ).otherwise(F.lit(0.0).cast("double"))
)

# Process invoice dataset
processed_invoices.collect()

job.commit()
