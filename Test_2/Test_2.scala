

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors

// The dataFrame is loaded from a csv
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")


//2 Columns
data.columns



// The schema of the dataFrame data is printed
data.printSchema()

// null fields are removed
val dataClean = data.na.drop()


// We see the first 5 data and observe that the DataFrame does not have adequate headers
data.show(5)

data.describe().show()

//  A vector is generated that contains the characteristics to be evaluated
// and are saved in the features column
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))

// The features are transformed using the dataframe
val features = vectorFeatures.transform(dataClean)

// 8Transform categorical data from species to numerical data with the label column
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")

// We adjust the indexed species with the vector features
val dataIndexed = speciesIndexer.fit(features).transform(features)

// Separate training data and testing data using indexed data
val splits = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)

// Set the layer settings for the model
val layers = Array[Int](4, 5, 4, 3)

// The Multilayer algorithm trainer is configured
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// Train the model using the training data
val model = trainer.fit(train)

// Run the model with the test data
val result = model.transform(test)

// The prediction and the label are selected
val predictionAndLabels = result.select("prediction", "label")

// Estimate model precision
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

// The result of the precision is printed
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")