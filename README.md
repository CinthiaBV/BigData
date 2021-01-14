# Big Data - development 
# Project

### <p align="center" > TECNOLÓGICO NACIONAL DE MÉXICO INSTITUTO TECNOLÓGICO DE TIJUANA SUBDIRECCIÓN ACADÉMICA DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN PERIODO: Agosto-Diciembre  2020</p>

###  <p align="center">  Carrera: Ing. En Sistemas Computacionales. 
### <p align="center"> Materia: 	Datos Masivos (BDD-1704 SC9A	).</p>

### <p align="center">  Maestro: Jose Christian Romero Hernandez	</p>
### <p align="center">  No. de control y nombre del alumno: 15211916 - Vargas Garcia Cinthia Gabriela</p>
### <p align="center">  No. de control y nombre del alumno: 16210561 - Oliver Cardenas Jesus Alejandro</p>

### Index


&nbsp;&nbsp;&nbsp;[Algorithm SMV](#SMV-algorithm)

&nbsp;&nbsp;&nbsp;[Algorithm DecisionThree](#DecisionThree-algorithm)

&nbsp;&nbsp;&nbsp;[Algorithm Logistic Regresion](#LogisticRegresion-algorithm)

&nbsp;&nbsp;&nbsp;[Algorithm Multilayer perceptron](#Multilayerperceptron-algorithm)

&nbsp;&nbsp;&nbsp;[Document](#Document)
  

&nbsp;&nbsp;&nbsp;[Link_Video](#Link-video)     

### &nbsp;&nbsp;Project .

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
// Project content

//1.- Objective: Comparison of the performance following machine learning algorithms

// - SVM

// - Decision Three

// - Logistic Regresion

// - Multilayer perceptron

// With the following data set: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

// Content of the final project document

// 1. Cover

// 2. Index

// 3. Introduction

// 4. Theoretical framework of algorithms

// 5. Implementation (What tools used and why (in this case spark with scala))

// 6. Results (A table with the data for each algorithm to see its performance)
// and its respective explanation.

// 7. Conclusions

// 8. References (No wikipedia for any reason, try to make them scientific articles)

// The document must be referenced

// Note: if the document is not presented, I will not review its development of the project


### &nbsp;&nbsp;Algorithm SMV.

        
#### In this practice, we did  the Algorithm SMV
     
</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala   
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
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


// The data is loaded into the variable "data" in the format "libsvm"
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

//Indexamos las columnas label y features
val dataIndexed = featuresLabel.select("label","features")


// We index the labels
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed) // features with > 4 distinct values are treated as continuous.



// We divide the data, 70% training and 30% testing
// val Array (trainingData, testData) = dataIndexed.randomSplit (Array (0.7, 0.3))
val splits = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)

// We create the model and indicate the columns to use
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

val lsvcModel = lsvc.fit(test)
// We put everything in a pipe

val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lsvc))


// Now if we train the model with 70% of the data
val model = pipeline.fit(trainingData)

// We perform the prediction of the data with 30% of the data

val predictions = model.transform(testData)


// let's print the first s5 records

predictions.select("prediction", "label", "features").show(5)

val predictionAndLabels = predictions.select("prediction", "label")
 
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
 
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
println("Accuracy = " + accuracy)
 
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

// 1mb -> 1e6 bytes
var mb = 0.000001
println("Used Memory: " + ((runtime.totalMemory - runtime.freeMemory) * mb) + " mb")
println("Free Memory: " + (runtime.freeMemory * mb) + " mb")
println("Total Memory: " + (runtime.totalMemory * mb) + " mb")
println("Max Memory: " + (runtime.maxMemory * mb)+ " mb")


val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (endTimeMillis - startTimeMillis) / 1000 + "s"


``` 
### &nbsp;&nbsp;Algorithm DecisionThree.

        
#### In this practice, we did  the Algorithm DecisionThree
     
</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala 
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


``` 
### &nbsp;&nbsp;Algorithm Logistic Regresion.

        
#### In this practice, we did  the Algorithm Logistic Regresion
</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala   
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression

// We minimize errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// We create a spark session
val spark = SparkSession.builder().getOrCreate()

//Load our  CSV file

val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

// Print schema
df.printSchema()

// We visualize our Dataframe
df.show()


// We modify the column of strings to numeric data
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
// We display the new column
newcolumn.show()

// We generate the features table
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
// We show the new column
fea.show()
// We change the column y to the label column
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
// Logistic Regression algorithm
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
// Fit of the model
val logisticModel = logistic.fit(feat)
// Impression of coefficients and interception
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")



``` 
### &nbsp;&nbsp;Algorithm Multilayer perceptron.

        
#### In this practice, we did  the Algorithm Multilayer perceptron
     
</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala   
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// We minimize errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// We create a spark session
val spark = SparkSession.builder().getOrCreate()

// load our CSV file

val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

// Print schema
df.printSchema()
// We visualize our Dataframe

df.show()



// We modify the column of strings to numeric data

val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))

//We generate the features table
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)

// We change the column to label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")

//Multilayer perceptron
// We divide the data in an array into parts of 70% and 30%
val split = feat.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = split(0)
val test = split(1)

// We specify the layers for the neural network
val layers = Array[Int](5, 2, 2, 4)


// We create the trainer with its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// We train the model
val model = trainer.fit(train)

// We print the accuracy
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")



``` 
## &nbsp;&nbsp;Document.

        

     
</br>

