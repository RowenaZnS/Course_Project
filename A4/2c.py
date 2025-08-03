import time
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_date, year, floor, mean, expr
from pyspark.sql.types import IntegerType, StructType, StringType, StructField, DoubleType
import pandas as pd

@udf(returnType=IntegerType())
def jdn(dt):
    """
    Computes the Julian date number for a given date.
    Parameters:
    - dt, datetime : the Gregorian date for which to compute the number

    Return value: an integer denoting the number of days since January 1,
    4714 BC in the proleptic Julian calendar.
    """
    y = dt.year
    m = dt.month
    d = dt.day
    if m < 3:
        y -= 1
        m += 12
    a = y // 100
    b = a // 4
    c = 2 - a + b
    e = int(365.25 * (y + 4716))
    f = int(30.6001 * (m + 1))
    jd = c + d + e + f - 1524
    return jd

def lsq(key, df):
    """
    Compute the simple linear regression with least squares using applyInPandas().
    """
    x = df['JDN'].values
    y = df['TAVG'].values
    x_bar = x.mean()
    y_bar = y.mean()
    dx = x - x_bar
    dy = y - y_bar
    beta = (dx * dy).sum() / (dx ** 2).sum()
    return pd.DataFrame({
        'STATION': [key[0]],
        'NAME': [key[1]],
        'BETA': [beta]
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute climate data.')
    parser.add_argument('-w', '--num-workers', default=1, type=int,
                        help='Number of workers')
    parser.add_argument('filename', type=str, help='Input filename')
    args = parser.parse_args()

    # Spark session initialization
    spark = SparkSession.builder \
        .master(f'local[{args.num_workers}]') \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()

    start = time.time()

    # Step 1: Read the data
    read_start = time.time()
    df = spark.read.options(delimiter=",", header=True).csv(args.filename)
    df = df.withColumn("DATE", to_date("DATE"))
    df = df.withColumn("JDN", jdn(col("DATE")))
    df = df.withColumn("TAVG", (col("TMIN") + col("TMAX")) / 2)
    read_end = time.time()

    # Step 2: Compute regression coefficients (BETA)
    schema = StructType([
        StructField("STATION", StringType()),
        StructField("NAME", StringType()),
        StructField("BETA", DoubleType()),
    ])
    beta_df = df.groupBy("STATION", "NAME").applyInPandas(lsq, schema).cache()
    top5_slopes = beta_df.orderBy(col("BETA").desc()).limit(5).collect()

    print('Top 5 coefficients:')
    for row in top5_slopes:
        print(f'{row["STATION"]} at {row["NAME"]} BETA={row["BETA"]:0.3e} °F/d')

    # Fraction of positive coefficients
    total = beta_df.count()
    positive = beta_df.filter(col("BETA") > 0).count()
    print('Fraction of positive coefficients:')
    print(positive / total)

    # Five-number summary of BETA values
    quantiles = beta_df.approxQuantile("BETA", [0.0, 0.25, 0.5, 0.75, 1.0], 0.001)
    beta_min, beta_q1, beta_median, beta_q3, beta_max = quantiles
    print('Five-number summary of BETA values:')
    print(f'beta_min {beta_min:0.3e}')
    print(f'beta_q1 {beta_q1:0.3e}')
    print(f'beta_median {beta_median:0.3e}')
    print(f'beta_q3 {beta_q3:0.3e}')
    print(f'beta_max {beta_max:0.3e}')

    # Step 3: Compute decade-wise temperature differences
    df = df.withColumn("DECADE", (floor(year("DATE") / 10) * 10).cast("int"))
    df_decade = df.filter(col("DECADE").isin([1910, 2010]))
    decade_avg = df_decade.groupBy("STATION", "NAME", "DECADE").agg(mean("TAVG").alias("TAVG_MEAN"))

    pivoted = decade_avg.groupBy("STATION", "NAME").pivot("DECADE", [1910, 2010]).agg(expr("first(TAVG_MEAN)"))
    pivoted = pivoted.withColumnRenamed("1910", "TAVG_1910").withColumnRenamed("2010", "TAVG_2010")
    pivoted = pivoted.dropna()

    pivoted = pivoted.withColumn("TAVG_DIFF", (col("TAVG_2010") - col("TAVG_1910")) * 5 / 9)

    # Top 5 differences
    top5_diffs = pivoted.orderBy(col("TAVG_DIFF").desc()).limit(5).collect()
    print('Top 5 differences:')
    for row in top5_diffs:
        print(f'{row["STATION"]} at {row["NAME"]} difference {row["TAVG_DIFF"]:0.1f} °C')

    # Fraction of positive differences
    diff_total = pivoted.count()
    diff_pos = pivoted.filter(col("TAVG_DIFF") > 0).count()
    print('Fraction of positive differences:')
    print(diff_pos / diff_total)

    # Five-number summary of temperature differences
    diff_quantiles = pivoted.approxQuantile("TAVG_DIFF", [0.0, 0.25, 0.5, 0.75, 1.0], 0.001)
    tdiff_min, tdiff_q1, tdiff_median, tdiff_q3, tdiff_max = diff_quantiles
    print('Five-number summary of decade average difference values:')
    print(f'tdiff_min {tdiff_min:0.1f} °C')
    print(f'tdiff_q1 {tdiff_q1:0.1f} °C')
    print(f'tdiff_median {tdiff_median:0.1f} °C')
    print(f'tdiff_q3 {tdiff_q3:0.1f} °C')
    print(f'tdiff_max {tdiff_max:0.1f} °C')

    # Total time measurement
    print(f'num workers: {args.num_workers}')
    print(f'Time spent reading data: {read_end - read_start:0.1f} s')
    print(f'Total time: {time.time() - start:0.1f} s')
