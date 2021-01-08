// import logistic regresion
//import Spark session
//Libraries
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//COnstruct Spark session
val spark = SparkSession.builder().getOrCreate()
//Access File advertising.csv
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")

//Deploy Data
data.printSchema()
//Print 1 row
data.head(1)

val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}


///Creation of new column Hour
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
//Rename Column CLicked on ad to Label
//Features elements
val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")


//Import Vector Assembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
//Create new object Vector Asssembler
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
                  .setOutputCol("features"))
//Random split
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

//Import Pipeline
import org.apache.spark.ml.Pipeline
//Create object lOgistic regresion
val lr = new LogisticRegression()
//Create new pipeline
val pipeline = new Pipeline().setStages(Array(assembler, lr))
//adjustment pipeline 
val model = pipeline.fit(training)

val results = model.transform(test)

//Import multiclassMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)
// print confussion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
//Metrics Accuracy
metrics.accuracy
