# Unit_2

### <p align="center" > TECNOLÓGICO NACIONAL DE MÉXICO INSTITUTO TECNOLÓGICO DE TIJUANA SUBDIRECCIÓN ACADÉMICA DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN PERIODO: Agosto-Diciembre  2020</p>

###  <p align="center">  Carrera: Ing. En Sistemas Computacionales. 
### <p align="center"> Materia: 	Datos Masivos (BDD-1704 SC9A	).</p>

### <p align="center">  Maestro: Jose Christian Romero Hernandez	</p>
### <p align="center">  No. de control y nombre del alumno: 15211916 - Vargas Garcia Cinthia Gabriela</p>

### Index
&nbsp;&nbsp;&nbsp;[Practice 1](#practice-1)  
&nbsp;&nbsp;&nbsp;[Practice 2](#practice-2)  
&nbsp;&nbsp;&nbsp;[Practice 3](#practice-3)    
&nbsp;&nbsp;&nbsp;[Practice 4](#practice-4)    
&nbsp;&nbsp;&nbsp;[Practice 5](#practice-5)  
&nbsp;&nbsp;&nbsp;[Practice 6](#practice-6)
&nbsp;&nbsp;&nbsp;[Practice 7](#practice-7)    
&nbsp;&nbsp;&nbsp;[Practice 8](#practice-8)    
&nbsp;&nbsp;&nbsp;[Practice 9](#practice-9)  
&nbsp;&nbsp;&nbsp;[Practice 10](#practice-10)    

### &nbsp;&nbsp;Practice 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
1. Import the LinearRegression
2. Use the following code to configure errors
3. Start a simple Spark Session
4. Use Spark for the Clean-Ecommerce csv file
5. Print the schema on the DataFrame
6. Print an example row from the DataFrame
7. Transform the data frame so that it takes the form of ("label", "features")
8. Rename the Yearly Amount Spent column as "label"
9. The VectorAssembler Object 
10. Use the assembler to transform our DataFrame to two columns: label and features 
11. Create an object for line regression model 
12. Fit the model for the data and call this model lrModel
13. Print the coefficients and intercept for the linear regression
14. Summarize the model on the training set and print the output of some metrics
15. Show the residuals values, the RMSE, the MSE, and also the R^2

#### In this practice  we use LinealRegression
 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.


```scala  
//1. Import LinearRegression
 
  1. import org.apache.spark.ml.regression.LinearRegression
//2. Use the following code to configure errors   	 
  2.import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//Start a simple Spark Session
3.import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
//Use Spark for the Clean-Ecommerce csv file 
  4. val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")
// Print the schema on the DataFrame
5. data.printSchema
//Print an example row from the DataFrame 
6. data.head(1)
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(0, colnames.length)){
   println(colnames(ind))
   println(firstrow(ind))
   println("\n")
}
//Transform the data frame so that it takes the form of ("label", "features")
7.   Import VectorAssembler and Vectors:
 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
	
//Rename the Yearly Amount Spent column as "label"
8. val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership", $"Yearly Amount Spent")
//The VectorAssembler Object 
9. val new_assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership", "Yearly Amount Spent")).setOutputCol("features")
//Use the assembler to transform our DataFrame to two columns: label and features 	
10. val output = new_assembler.transform(df).select($"label",$"features")
//Create an object for line regression model 
11. val lr = new LinearRegression()
 //Fit the model for the data and call this model lrModel
12.val lrModel = lr.fit(output)
//Print the coefficients and intercept for the linear regression
13. println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
//Summarize the model on the training set and print the output of some metrics
14.val trainingSummary = lrModel.summary
 
 //Show the residuals values, the RMSE, the MSE, and also the R^2
15. trainingSummary.residuals.show()
val RMSE = trainingSummary.rootMeanSquaredError
val MSE = scala.math.pow(RMSE, 2.0)
val R2 = trainingSummary.r2 
```

### &nbsp;&nbsp;Practice 2.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
1. Import logistic regresion
2. Import Spark session
3. Libraries
4. Construct Spark session
5. Access File advertising.csv
6. Deploy Data
7. Creation of new column Hour
8. Rename Column CLicked on ad to Label
9. Features elements
10. Print 1 row
11. Creation of new column Hour
12. Rename Column CLicked on ad to Label
13. Features elements
14. Import Vector Assembler and Vectors
15. Create new object Vector Asssembler
16. Random split
17. Import Pipeline
18. Create object lOgistic regresion
19. Create new pipeline
20. adjustment pipeline 
21. Import multiclassMetrics
22. Print confussion matrix
23. Metrics Accuracy

#### In this practice  we use LogisticRegression

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala 
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
```

### &nbsp;&nbsp;Practice 3.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

1. Import linalg 
2. Import stat
3. Import sql
4. Correlation
5. A value called data is created to which a sequence of vectors is assigned as value
6. A dataframe is created to which the value of a Tuple called Tuple1 is assigned,
7. The dataframe contains a column called feautures
8. A Row type value called coefficient1 of a matrix is ​​assigned the value of the pearson correlation applied to the dataframe
9. Applied to its features column
10. Print 

#### In this practice  we use BasicStadistics

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala 

import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

//Correlation
object CorrelationExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("CorrelationExample")
      .getOrCreate()
    import spark.implicits._

    // A value called data is created to which a sequence of vectors is assigned as value
    val data = Seq(
      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))), // (0,3,1,-2)
      Vectors.dense(4.0, 5.0, 0.0, 3.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0),
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0))) // (0,3,9,1)
    )

    // A dataframe is created to which the value of a Tuple called Tuple1 is assigned,
    // The dataframe contains a column called feautures
    val df = data.map(Tuple1.apply).toDF("features")
    // A Row type value called coefficient1 of a matrix is ​​assigned the value of the pearson correlation applied to the dataframe
    // Applied to its features column
    val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
    println(s"Pearson correlation matrix:\n $coeff1") //Se imprime
    // A Row type value called coefficient2 of a matrix is ​​assigned the value of the spearman correlation applied to the dataframe
    // Applied to its features column
    val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
    //Print 
    println(s"Spearman correlation matrix:\n $coeff2") 
    

    spark.stop()
  }
}

```

### &nbsp;&nbsp;Practice 4.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

1. Import libraries.
2. Import a Spark Session.
3. Create a Spark session.
4. Load the data stored in LIBSVM format as a DataFrame.
5. Index labels, adding metadata to the label column.
6. Automatically identify categorical features, and index them.
7. Split the data into training and test sets.
8. Train a DecisionTree model. 
9. Convert indexed labels back to original labels.
10. Chain indexers and tree in a Pipeline.
11. Train model.
12. Make predictions.
13. Select example rows to display.
14. Select (prediction, true label) and compute test error.
15. Print the tree obtained from the model

#### In this practice  we use Decision tree clasifier


#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

  	 
``` scala
//Import libraries.
1. import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
 
//Import a Spark Session.
2.import org.apache.spark.sql.SparkSession

//Create a Spark session.
3. def main(): Unit = {
   val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

//Load the data stored in LIBSVM format as a DataFrame.
4.    val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
5. val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

//Automatically identify categorical features, and index them.
6. 
 val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

//Split the data into training and test sets.
 
7.    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

//Train a DecisionTree model. 
8.    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
 
//Convert indexed labels back to original labels. 
9.   val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)


//Chain indexers and tree in a Pipeline.
10   val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
 
// Train model.
11. 
   val model = pipeline.fit(trainingData)
 
// Make predictions.
12.    val predictions = model.transform(testData)
 
//Select example rows to display. 
13.    predictions.select("predictedLabel", "label", "features").show(5)
 
//Select (prediction, true label) and compute test error. 
14. 
   val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
   val accuracy = evaluator.evaluate(predictions)
   println(s"Test Error = ${(1.0 - accuracy)}")

//   Print the tree obtained from the model

15.    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
   println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
 ```
 
 ### &nbsp;&nbsp;Practice 5.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.


1. Import libraries.
2. Import a Spark Session.
3. Create a Spark session.
4. Load the data and create a Dataframe.
5. Index labels
6. Automatically identify categorical features, and index them.
7. Split the data
8. Train a RandomForest mode
9. Convert indexed labels.
10. Chain indexers and forest in a Pipeline.
11. Train model.
12. Make predictions.
13. Select example rows to display.
14. Select (prediction, true label) and compute test error.
15. Print the trees obtained from the model.

#### In this practice  we use Random forest classifier

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

``` scala
// Import libraries.
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

//Import a Spark Session.
 import org.apache.spark.sql.SparkSession
//Create a Spark session.
def main(): Unit = {
   val spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()
// Load the data and create a Dataframe.
 val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
 //Index labels 
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
 //Automatically identify categorical features, and index them.   
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
 // Split the data    
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
 // Train a RandomForest mode   
    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees

 // Convert indexed labels.   
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

 // Chain indexers and forest in a Pipeline.   
   val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
 // Train model   
   val model = pipeline.fit(trainingData)

 // Make predictions   
   val predictions = model.transform(testData)

 //  Select example rows to display.
   predictions.select("predictedLabel", "label", "features").show(5)

 //  Select (prediction, true label) and compute test error.
   val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
   val accuracy = evaluator.evaluate(predictions)
   println(s"Test Error = ${(1.0 - accuracy)}")

// Print the trees obtained from the model.   
   val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
   println(s"Learned classification forest model:\n ${rfModel.toDebugString}")

   ```

### &nbsp;&nbsp;Practice 6.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

1. Import Libraries
2. Import Pipeline
3. Import Gradient Classification MOdel
4. Import MulticlassClasificatioon
5. Load and parse the data file, converting it to a DataFrame.
6. Index labels, adding metadata to the label column.
7. Fit on whole dataset to include all labels in index.
8. Automatically identify categorical features, and index them.
9. Set maxCategories so features with > 4 distinct values are treated as continuous.
10. Split the data into training and test sets (30% held out for testing).
11. Train a GBT model. Set labelCol as indexed label 
12. Setr feature col
13. Max number of iterations
14. Convert indexed labels back to original labels.
15. Chain indexers and GBT in a Pipeline.
16. Label Converter
17. Train model. This also runs the indexers.
18. Select example rows to display.
19. Show Rows
20. Select (prediction, true label) and compute test error.
21. Print accuracy
22. Print Learned Classification Model

#### In this practice  we use Gradient-boosted 
 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

``` scala

 ///Import Libraries
//Import Pipeline
//Import Gradient Classification MOdel
//Import MulticlassClasificatioon
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
// Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a GBT model. Set labelCol as indexed label 
//Setr feature col
//Max number of iterations
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and GBT in a Pipeline.
//Label Converter
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
//Show Rows
predictions.select("predictedLabel", "label", "features").show(20)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
//Print accuracy
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
//Print Learned Classification Model
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
 ```

 ### &nbsp;&nbsp;Practice 7.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.


#### In this practice  we use Multilayer perceptron classifier
 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.


```scala
// Se importa MultilayerPerceptronClassifier y MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// $example off$
// Se impor ta la sesion de Spark
import org.apache.spark.sql.SparkSession

/**
 * An example for Multilayer Perceptron Classification.
 */

 //Creation of the MultilayerPerceptronClassifier object
object MultilayerPerceptronClassifierExample
object MultilayerPerceptronClassifierExample {

// The main function is defined which has as a parameter an Array of type string
  def main(): Unit = {
    // The SparkSession class object is created, and the app is given the name of
    // MultilayerPerceptronClassifierExample
    val spark = SparkSession
      .builder
      .appName("MultilayerPerceptronClassifierExample")
      .getOrCreate()

    // $example on$
    // The data in libsvm format is loaded from the file as a DataFrame
    val data = spark.read.format("libsvm")
      .load("sample_multiclass_classification_data.txt")

    // Divide the data into training and testing
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // The layers of the neural network are specified:
    // The input layer is size 4 (characteristics), two intermediate layers
    // one of size 5 and the other of size 4
    val layers = Array[Int](4, 5, 4, 3)

    // The training parameters are set
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // The model is trained
    val model = trainer.fit(train)

    // The precision of the test data is calculated
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

     // The exactitude of the model is printed
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
    // $example off$

    spark.stop()
  }
}


 ```
  

  