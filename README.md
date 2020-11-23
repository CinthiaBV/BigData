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
 