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