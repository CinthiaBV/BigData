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
 