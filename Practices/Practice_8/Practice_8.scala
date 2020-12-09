//1. Import libraries and package
1.package org.apache.spark.examples.ml
import org.apache.spark.ml.classification.LinearSVC
// Import a Spark Session. 3.Load the data from the file and add it to a variable to train it.
2.import org.apache.spark.sql.SparkSession
//Load the data stored in LIBSVM format as a DataFrame.
3.val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

4. val training = spark.read.format("libsvm").load("/usr/local/spark-2.3.4-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")
5. val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
6. val lsvcModel = lsvc.fit(training)
7. println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
