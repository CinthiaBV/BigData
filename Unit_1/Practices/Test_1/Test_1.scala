//1. Start a simple Spark session.

import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()


//2. Upload Netflix Stock CSV file, have Spark infer data types. 

val df =spark.read.option("header","true").option("inferSchema","true").csv("Netflix_2011_2016.csv")

//3. What are the names of the columns?

df.columns
//4. What is the scheme like?

df.printSchema()
//5. Print the first 5 columns.

df.head(5)
//6. Use describe () to learn about the DataFrame.

df.describe().show()

//7.Create a of dataframe with column called "HV Ratio" which is the
//relationship between the price of the column "High" versus the column "Volume" of
//shares traded for one day.
val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))

//8.What day had the highest peak in the “Close” column?
df.orderBy($"High".desc).show(1)

//9.for me column close represents the average of the document
df.select(mean("Close")).show()

//10. What is the max and min of the Volume column?
df.select(max("Volume")).show()
df.select(min("Volume")).show()

// For Scala/Spark $ Syntax
import spark.implicits._

//11a. How many days was the Close lower than $ 600?
df.filter($"Close"<600).count()

//11.b What percentage of the time was the High greater than $500 ?
(df.filter($"High">500).count()*1.0/df.count())*100

//11.c What is the Pearson correlation between High and Volume?
df.select(corr("High","Volume")).show()

//11.d What is the max High per year?
val yeardf = df.withColumn("Year",year(df("Date")))
val yearmaxs = yeardf.select($"Year",$"High").groupBy("Year").max()
yearmaxs.select($"Year",$"max(High)").show()

//11.e What is the average Close for each Calender Month?
val monthdf = df.withColumn("Month",month(df("Date")))
val monthavgs = monthdf.select($"Month",$"Close").groupBy("Month").mean()
monthavgs.select($"Month",$"avg(Close)").show()






