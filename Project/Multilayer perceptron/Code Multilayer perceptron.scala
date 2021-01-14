
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

