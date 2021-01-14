//Import Data frames
import spark.implicits._

val Information = Seq (
    ("Cinthia","Sales","B.C", 45,2,2333),
    ("Laura","HR","B.C", 28,21,3333),
    ("Karla","QC","B.C", 45,20,6333),
    ("Hugo","QC","B.C", 35,4,6333),
    ("Anabel","Sales","B.C", 21,54,7899)
)
val df= simpleData.toDF("name","area","city","age","hours","payment")

//1 Show the information of Dataframe
df.show()

//2 Returns the population covariance for two columns.
df.select(covar_pop("Sales","Sales")).show()

//3 Returns the sample covariance for two columns.

df.select(covar_samp("Sales","Sales2")).show()

//4 Returns the first element in a column.
df.select(first("Sales")).show()


//5 Select only the last item of Sales

df.select(last("Sales")).show()

//6 Returns the sum of all distinct values in a column.
df.select(sumDistinct("Sales")).show()

// 7 Returns the population covariance for two columns.

df.select(covar_pop("Sales")).show()

//8 Function returns the average of values in the input column.
df.select(avg("Sales")).show()


// 9 Returns the unbiased variance of the values in a column.
 
df.select(var_samp("Sales")).show()

//10 Returns the count of distinct items in a group.
df.select(approx_count_distinct("Sales")).show()


//11 Returns all values from an input column with duplicates.
df.select(collect_list("Sales")).show()
 

// 12 Returns the kurtosis of the values in a group. 
df.select(kurtosis("Sales")).show()

//13 Returns the skewness of the values in a group. 
df.select(skewness("Sales")).show()


//14 Returns the maximum value in a column.
df.select(max("Company")).show()

// 15 Returns the Pearson Correlation Coefficient for two columns.
df.select(corr("Sales","Sales2")).show()
