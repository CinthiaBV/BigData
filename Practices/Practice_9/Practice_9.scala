1.import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
2.import org.apache.spark.sql.SparkSession

3. def main(): Unit = {
   val spark = SparkSession.builder.appName("MulticlassClassificationEvaluator").getOrCreate()
4.val inputData = spark.read.format("libsvm")load("data/mllib/sample_multiclass_classification_data.txt")
5.val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
 
 6.val classifier = new LogisticRegression()
.setMaxIter(10)
.setTol(1E-6)
.setFitIntercept(true)

7val ovr = new OneVsRest().setClassifier(classifier)

8. val ovrModel = ovr.fit(train)
  
9. val predictions = ovrModel.transform(test)

10. val evaluator = new MulticlassClassificationEvaluator()
.setMetricName("accuracy")
11.val accuracy = evaluator.evaluate(predictions)
12.println(s"Test Error = ${1 - accuracy}")