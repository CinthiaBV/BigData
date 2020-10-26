//1. Start a simple Spark session.

import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()


//2. Upload Netflix Stock CSV file, have Spark infer data types. 

val df =spark.read.option("header","true").option("inferSchema","true").csv("Netflix_2011_2016.csv")

//3. What are the names of the columns?

df.columns
//4. What is the scheme like?

//df.printSchema(5)
//5. Print the first 5 columns.

//df.head(5)
//6. Use describe () to learn about the DataFrame.

//df.describe().show()


