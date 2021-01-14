import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder,IndexToString}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

// Minimize errors shown
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// Values ​​to measure performance
val runtime = Runtime.getRuntime
val startTimeMillis = System.currentTimeMillis()



// Start a session in spark
val spark = SparkSession.builder().getOrCreate()


// The data is loaded in the variable "data" in the format "csv"
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

// Print Schema
data.printSchema()


// We convert the values ​​of column "y" into numeric
val change1 = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),0).otherwise(col("y")))
val newDF = change2.withColumn("y",'y.cast("Int"))


// We generate a vector with the names of the columns to evaluate

val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))

// The vector is filled with the values
val features = vectorFeatures.transform(newDF)

// We rename the column "y" to Label
val featuresLabel = features.withColumnRenamed("y", "label")

// We index the label and features columns
val dataIndexed = featuresLabel.select("label","features")

// We index the labels
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed) 

// We divide the data, 70% training and 30% testing
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))

// We create the model and indicate the columns to use
val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")

// We put everything in a pipe
val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))

// Now if we train the model with 70% of the data
val model = pipeline.fit(trainingData)

// We perform the prediction of the data with 30% of the data
val predictions = model.transform(testData)

// let's print the first s5 records
predictions.select("prediction", "label", "features").show(5)

// We select the columns and the value of the error
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println("Learned regression tree model:\n" + treeModel.toDebugString)

// 1mb -> 1e6 bytes
var mb = 0.000001
println("Used Memory: " + ((runtime.totalMemory - runtime.freeMemory) * mb) + " mb")
println("Free Memory: " + (runtime.freeMemory * mb) + " mb")
println("Total Memory: " + (runtime.totalMemory * mb) + " mb")
println("Max Memory: " + (runtime.maxMemory * mb)+ " mb")

val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (endTimeMillis - startTimeMillis) / 1000 + "s"
