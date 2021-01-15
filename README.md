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

  <div align="justify">  


Contenido

Introducción	3
Marco Teórico	4
Máquinas de vectores de soporte	4
Árbol de decisión	4
Regresión logística	4
Perceptrón multicapa	4
Implementación	5
Herramientas Utilizadas	5
Resultados	6
Corrida1 Perceptron Multicapa	6
Corrida 1 Logistic Regresssion	10
Corrida 1 Arbol de Decision	16
Corrida 1  Maquina de vectores	20
Conclusiones	22
Referencias bibliográficas	23
Anexos	24
Code Support Vector Machines	24
Code Decision Tree	26
Code Logistic Regression	27
Code Multilayer perceptron	29









Introducción

Como idea principal, podemos decir que el aprendizaje automático (Machine Learning) es la ciencia de los algoritmos que se encarga de darle sentido a los datos. Este concepto es adecuado indicarlo pues cada vez estamos rodeados de más información de la que creemos, algo que si sabemos procesarlo y hacemos un adecuado uso de los algoritmos de autoaprendizaje nos puede ayudar a identificar patrones y con ello clasificar o predecir partiendo de nuevos conjuntos de datos que no han sido tratados previamente. Esta técnica se basa en el uso datos estadísticos, probabilísticos y de técnicas de optimización para dar la capacidad de aprender a las máquinas por sí solas. Existen gran cantidad de algoritmos de aprendizaje automático que son usados para llevar a cabo tareas de clasificación o de diagnóstico.
Para poder llevar a cabo lo anteriormente mencionado es necesario: proveer los datos, almacenarlos y, finalmente, procesarlos para obtener los resultados que buscamos.
En este proyecto que realizamos en conjunto mi  compañera Cinthia y yo su compañero Alejandro en la ya conocida técnica de pair coding , probaremos la eficiencia de diferentes algoritmos a unos datos en concreto para poder probar cual posee la mayor eficiencia  en resolverlo. Los algoritmos que se utilizaran serán los siguientes en la prueba de rendimiento: Máquina de vectores de soporte, Árbol de Decisión, Regresión Logística, Perceptrón Multicapa.

Corremos cada método 10 veces y sacaremos el promedio.  Después de  los datos obtenidos  podemos generar tablas donde almacenaremos los valores que cada clasificador ha generado para así dibujar gráficas comparativas e indicar cuál sería la mejor opción a elegir para tratar el problema en el que estamos implicados. Podemos definir que queremos obtener una exactitud cercana al 100% además de un error de predicción bajo, valores que nos indicarán una confianza mayor en dicho clasificador para aquellos datos suministrados que no hayan sido observados previamente. 












Marco Teórico

Máquinas de vectores de soporte

En el aprendizaje automático, las máquinas de vectores de apoyo (SVM, también redes de vectores de apoyo) son modelos de aprendizaje supervisados con algoritmos de aprendizaje asociados que analizan los datos utilizados para el análisis de clasificación y regresión.

Árbol de decisión

Un árbol de decisiones es una herramienta de apoyo a las decisiones que utiliza un modelo de decisiones en forma de árbol y sus posibles consecuencias, incluidos los resultados de eventos fortuitos, los costos de los recursos y la utilidad. Es una forma de mostrar un algoritmo que solo contiene declaraciones de control condicionales.

Los árboles de decisión se usan comúnmente en la investigación de operaciones, específicamente en el análisis de decisiones, para ayudar a identificar una estrategia con más probabilidades de alcanzar un objetivo, pero también son una herramienta popular en el aprendizaje automático.

Regresión logística

En estadística, el modelo logístico (o modelo logit) se utiliza para modelar la probabilidad de que exista una determinada clase o evento, como pasa / no pasa, gana / pierde, vivo / muerto o sano / enfermo. Esto se puede ampliar para modelar varias clases de eventos, como determinar si una imagen contiene un gato, un perro, un león, etc. A cada objeto que se detecte en la imagen se le asignará una probabilidad entre 0 y 1, con una suma de uno.

Perceptrón multicapa

Este es uno de los tipos de redes más comunes. Se basa en otra red más simple llamada perceptrón simple, solo que el número de capas ocultas puede ser mayor o igual a uno. Es una red unidireccional (feedforward).


Implementación
Herramientas Utilizadas

Spark: 
Spark es framework que nos da muchas facilidades para trabajar con Big Data Spark SQL es un módulo Spark para el procesamiento de datos estructurados. A diferencia de la API básica de Spark RDD, las interfaces proporcionadas por Spark SQL brindan a Spark más información sobre la estructura tanto de los datos como del cálculo que se realiza. Internamente, Spark SQL usa esta información adicional para realizar optimizaciones adicionales. Hay varias formas de interactuar con Spark SQL, incluidas SQL y la API de conjunto de datos. Cuando se calcula un resultado, se utiliza el mismo motor de ejecución, independientemente de qué API / lenguaje se esté utilizando para expresar el cálculo. Esta unificación significa que los desarrolladores pueden alternar fácilmente entre diferentes API en función de cuál proporciona la forma más natural de expresar una transformación determinada.

Scala: 
Scala es un lenguaje de programación multiparadigma que es compatible con el framework Spark.
Scala es un lenguaje de programación con característica de paradigmas múltiples, en el cual se combina la programación orientada a objeto y todo lo que tenga relación con la programación funcional, intentando no dejar por fuera detalle alguno por pequeño que sea. La denominación de Scala proviene del significado Scalable Lenguaje, dicho nombre ha surgido por el propósito de un lenguaje capaz de crecer acorde a la demanda generada por los usuarios.
 
Una de las características atractivas de Scala es que puede ser usado para desarrollar pequeños scripts y transformarlos en sistemas de gran magnitud y sofisticado, lo que ha hecho de Scala un plus al momento de elegirlo.
Entre las principales características de Scala y que no pueden dejar de mencionarse, están:
•	Tipificado estadístico
•	Expresivo
•	Ligero
•	Alto nivel
•	Orientado a objetos
•	Cierres léxicos
•	Eficiencia
•	Conciso
•	Interoperatividad

Resultados 

Corrida1 Perceptrón Multicapa

df.printSchema()
root
 |-- age: integer (nullable = true)
 |-- job: string (nullable = true)
 |-- marital: string (nullable = true)
 |-- education: string (nullable = true)
 |-- default: string (nullable = true)
 |-- balance: integer (nullable = true)
 |-- housing: string (nullable = true)
 |-- loan: string (nullable = true)
 |-- contact: string (nullable = true)
 |-- day: integer (nullable = true)
 |-- month: string (nullable = true)
 |-- duration: integer (nullable = true)
 |-- campaign: integer (nullable = true)
 |-- pdays: integer (nullable = true)
 |-- previous: integer (nullable = true)
 |-- poutcome: string (nullable = true)
 |-- y: string (nullable = true)

scala> // We visualize our Dataframe

scala>

scala> df.show()
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|     	job| marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|  management| married| tertiary| 	no|   2143|	yes|  no|unknown|  5|  may| 	261|   	1|   -1|   	0| unknown| no|
| 44|  technician|  single|secondary| 	no| 	29|	yes|  no|unknown|  5|  may| 	151|   	1|   -1|   	0| unknown| no|
| 33|entrepreneur| married|secondary| 	no|  	2|	yes| yes|unknown|  5|  may|  	76|   	1|   -1|   	0| unknown| no|
| 47| blue-collar| married|  unknown| 	no|   1506|	yes|  no|unknown|  5|  may|  	92|   	1|   -1|   	0| unknown| no|
| 33| 	unknown|  single|  unknown| 	no|  	1| 	no|  no|unknown|  5|  may| 	198|   	1|   -1|   	0| unknown| no|
| 35|  management| married| tertiary| 	no|	231|	yes|  no|unknown|  5|  may| 	139|   	1|   -1|   	0| unknown| no|
| 28|  management|  single| tertiary| 	no|	447|	yes| yes|unknown|  5|  may| 	217|   	1|   -1|   	0| unknown| no|
| 42|entrepreneur|divorced| tertiary|	yes|  	2|	yes|  no|unknown|  5|  may| 	380|   	1|   -1|   	0| unknown| no|
| 58| 	retired| married|  primary| 	no|	121|	yes|  no|unknown|  5|  may|  	50|   	1|   -1|   	0| unknown| no|
| 43|  technician|  single|secondary| 	no|	593|	yes|  no|unknown|  5|  may|  	55|   	1|   -1|   	0| unknown| no|
| 41|  	admin.|divorced|secondary| 	no|	270|	yes|  no|unknown|  5|  may| 	222|   	1|   -1|   	0| unknown| no|
| 29|  	admin.|  single|secondary| 	no|	390|	yes|  no|unknown|  5|  may| 	137|   	1|   -1|   	0| unknown| no|
| 53|  technician| married|secondary| 	no|  	6|	yes|  no|unknown|  5|  may| 	517|   	1|   -1|   	0| unknown| no|
| 58|  technician| married|  unknown| 	no| 	71|	yes|  no|unknown|  5|  may|  	71|   	1|   -1|   	0| unknown| no|
| 57|	services| married|secondary| 	no|	162|	yes|  no|unknown|  5|  may| 	174|   	1|   -1|   	0| unknown| no|
| 51| 	retired| married|  primary| 	no|	229|	yes|  no|unknown|  5|  may| 	353|   	1|   -1|   	0| unknown| no|
| 45|  	admin.|  single|  unknown| 	no| 	13|	yes|  no|unknown|  5|  may|  	98|   	1|   -1|   	0| unknown| no|
| 57| blue-collar| married|  primary| 	no| 	52|	yes|  no|unknown|  5|  may|  	38|   	1|   -1|   	0| unknown| no|
| 60| 	retired| married|  primary| 	no| 	60|	yes|  no|unknown|  5|  may| 	219|   	1|   -1|   	0| unknown| no|
| 33|	services| married|secondary| 	no|  	0|	yes|  no|unknown|  5|  may|  	54|   	1|   -1|   	0| unknown| no|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 20 rows

scala>

scala>

scala>

scala> // We modify the column of strings to numeric data

scala>

scala> val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
change1: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]

scala> val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
change2: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]

scala> val newcolumn = change2.withColumn("y",'y.cast("Int"))
newcolumn: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]

scala>

scala> //We generate the features table

scala> val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_4e21326b4b57

scala> val fea = assembler.transform(newcolumn)
fea: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]

scala>

scala> // We change the column to label

scala> val cambio = fea.withColumnRenamed("y", "label")
cambio: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]

scala> val feat = cambio.select("label","features")
feat: org.apache.spark.sql.DataFrame = [label: int, features: vector]

scala>

scala> //Multilayer perceptron

scala> // We divide the data in an array into parts of 70% and 30%

scala> val split = feat.randomSplit(Array(0.7, 0.3), seed = 1234L)
split: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: int, features: vector], [label: int, features: vector])

scala> val train = split(0)
train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]

scala> val test = split(1)
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, features: vector]

scala>

scala> // We specify the layers for the neural network

scala> val layers = Array[Int](5, 2, 2, 4)
layers: Array[Int] = Array(5, 2, 2, 4)

scala>

scala>

scala> // We create the trainer with its parameters

scala> val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_8abf27ed26b2

scala>

scala> // We train the model

scala> val model = trainer.fit(train)
21/01/14 20:28:33 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/01/14 20:28:33 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
model: org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel = mlpc_8abf27ed26b2

scala>

scala> // We print the accuracy

scala> val result = model.transform(test)
result: org.apache.spark.sql.DataFrame = [label: int, features: vector ... 3 more fields]

scala> val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels: org.apache.spark.sql.DataFrame = [prediction: double, label: int]

scala> val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_a3ca17dfa9cb

scala> println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
Test set accuracy = 0.8862768145753747                  	


Corrida 1 Logistic Regresssion

// Print schema

scala> df.printSchema()
root
 |-- age: integer (nullable = true)
 |-- job: string (nullable = true)
 |-- marital: string (nullable = true)
 |-- education: string (nullable = true)
 |-- default: string (nullable = true)
 |-- balance: integer (nullable = true)
 |-- housing: string (nullable = true)
 |-- loan: string (nullable = true)
 |-- contact: string (nullable = true)
 |-- day: integer (nullable = true)
 |-- month: string (nullable = true)
 |-- duration: integer (nullable = true)
 |-- campaign: integer (nullable = true)
 |-- pdays: integer (nullable = true)
 |-- previous: integer (nullable = true)
 |-- poutcome: string (nullable = true)
 |-- y: string (nullable = true)

scala>

scala> // We visualize our Dataframe

scala> df.show()
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|     	job| marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|  management| married| tertiary| 	no|   2143|	yes|  no|unknown|  5|  may| 	261|   	1|   -1|   	0| unknown| no|
| 44|  technician|  single|secondary| 	no| 	29|	yes|  no|unknown|  5|  may| 	151|   	1|   -1|   	0| unknown| no|
| 33|entrepreneur| married|secondary| 	no|  	2|	yes| yes|unknown|  5|  may|  	76|   	1|   -1|   	0| unknown| no|
| 47| blue-collar| married|  unknown| 	no|   1506|	yes|  no|unknown|  5|  may|  	92|   	1|   -1|   	0| unknown| no|
| 33| 	unknown|  single|  unknown| 	no|  	1| 	no|  no|unknown|  5|  may| 	198|   	1|   -1|   	0| unknown| no|
| 35|  management| married| tertiary| 	no|	231|	yes|  no|unknown|  5|  may| 	139|   	1|   -1|   	0| unknown| no|
| 28|  management|  single| tertiary| 	no|	447|	yes| yes|unknown|  5|  may| 	217|   	1|   -1|   	0| unknown| no|
| 42|entrepreneur|divorced| tertiary|	yes|  	2|	yes|  no|unknown|  5|  may| 	380|   	1|   -1|   	0| unknown| no|
| 58| 	retired| married|  primary| 	no|	121|	yes|  no|unknown|  5|  may|  	50|   	1|   -1|   	0| unknown| no|
| 43|  technician|  single|secondary| 	no|	593|	yes|  no|unknown|  5|  may|  	55|   	1|   -1|   	0| unknown| no|
| 41|  	admin.|divorced|secondary| 	no|	270|	yes|  no|unknown|  5|  may| 	222|   	1|   -1|   	0| unknown| no|
| 29|  	admin.|  single|secondary| 	no|	390|	yes|  no|unknown|  5|  may| 	137|   	1|   -1|   	0| unknown| no|
| 53|  technician| married|secondary| 	no|  	6|	yes|  no|unknown|  5|  may| 	517|   	1|   -1|   	0| unknown| no|
| 58|  technician| married|  unknown| 	no| 	71|	yes|  no|unknown|  5|  may|  	71|   	1|   -1|   	0| unknown| no|
| 57|	services| married|secondary| 	no|	162|	yes|  no|unknown|  5|  may| 	174|   	1|   -1|   	0| unknown| no|
| 51| 	retired| married|  primary| 	no|	229|	yes|  no|unknown|  5|  may| 	353|   	1|   -1|   	0| unknown| no|
| 45|  	admin.|  single|  unknown| 	no| 	13|	yes|  no|unknown|  5|  may|  	98|   	1|   -1|   	0| unknown| no|
| 57| blue-collar| married|  primary| 	no| 	52|	yes|  no|unknown|  5|  may|  	38|   	1|   -1|   	0| unknown| no|
| 60| 	retired| married|  primary| 	no| 	60|	yes|  no|unknown|  5|  may| 	219|   	1|   -1|   	0| unknown| no|
| 33|	services| married|secondary| 	no|  	0|	yes|  no|unknown|  5|  may|  	54|   	1|   -1|   	0| unknown| no|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 20 rows

scala>

scala>

scala> // We modify the column of strings to numeric data

scala> val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
change1: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]

scala> val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
change2: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]

scala> val newcolumn = change2.withColumn("y",'y.cast("Int"))
newcolumn: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]

scala> // We display the new column

scala> newcolumn.show()
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
|age|     	job| marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
| 58|  management| married| tertiary| 	no|   2143|	yes|  no|unknown|  5|  may| 	261|   	1|   -1|   	0| unknown|  2|
| 44|  technician|  single|secondary| 	no| 	29|	yes|  no|unknown|  5|  may| 	151|   	1|   -1|   	0| unknown|  2|
| 33|entrepreneur| married|secondary| 	no|  	2|	yes| yes|unknown|  5|  may|  	76|   	1|   -1|   	0| unknown|  2|
| 47| blue-collar| married|  unknown| 	no|   1506|	yes|  no|unknown|  5|  may|  	92|   	1|   -1|   	0| unknown|  2|
| 33| 	unknown|  single|  unknown| 	no|  	1| 	no|  no|unknown|  5|  may| 	198|   	1|   -1|   	0| unknown|  2|
| 35|  management| married| tertiary| 	no|	231|	yes|  no|unknown|  5|  may| 	139|   	1|   -1|   	0| unknown|  2|
| 28|  management|  single| tertiary| 	no|	447|	yes| yes|unknown|  5|  may| 	217|   	1|   -1|   	0| unknown|  2|
| 42|entrepreneur|divorced| tertiary|	yes|  	2|	yes|  no|unknown|  5|  may| 	380|   	1|   -1|   	0| unknown|  2|
| 58| 	retired| married|  primary| 	no|	121|	yes|  no|unknown|  5|  may|  	50|   	1|   -1|   	0| unknown|  2|
| 43|  technician|  single|secondary| 	no|	593|	yes|  no|unknown|  5|  may|  	55|   	1|   -1|   	0| unknown|  2|
| 41|  	admin.|divorced|secondary| 	no|	270|	yes|  no|unknown|  5|  may| 	222|   	1|   -1|   	0| unknown|  2|
| 29|  	admin.|  single|secondary| 	no|	390|	yes|  no|unknown|  5|  may| 	137|   	1|   -1|   	0| unknown|  2|
| 53|  technician| married|secondary| 	no|  	6|	yes|  no|unknown|  5|  may| 	517|   	1|   -1|   	0| unknown|  2|
| 58|  technician| married|  unknown| 	no| 	71|	yes|  no|unknown|  5|  may|  	71|   	1|   -1|   	0| unknown|  2|
| 57|	services| married|secondary| 	no|	162|	yes|  no|unknown|  5|  may| 	174|   	1|   -1|   	0| unknown|  2|
| 51| 	retired| married|  primary| 	no|	229|	yes|  no|unknown|  5|  may| 	353|   	1|   -1|   	0| unknown|  2|
| 45|  	admin.|  single|  unknown| 	no| 	13|	yes|  no|unknown|  5|  may|  	98|   	1|   -1|   	0| unknown|  2|
| 57| blue-collar| married|  primary| 	no| 	52|	yes|  no|unknown|  5|  may|  	38|   	1|   -1|   	0| unknown|  2|
| 60| 	retired| married|  primary| 	no| 	60|	yes|  no|unknown|  5|  may| 	219|   	1|   -1|   	0| unknown|  2|
| 33|	services| married|secondary| 	no|  	0|	yes|  no|unknown|  5|  may|  	54|   	1|   -1|   	0| unknown|  2|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
only showing top 20 rows

scala>

scala> // We generate the features table

scala> val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_2ae21e7ac5ce

scala> val fea = assembler.transform(newcolumn)
fea: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]

scala> // We show the new column

scala> fea.show()
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
|age|     	job| marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|        	features|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
| 58|  management| married| tertiary| 	no|   2143|	yes|  no|unknown|  5|  may| 	261|   	1|   -1|   	0| unknown|  2|[2143.0,5.0,261.0...|
| 44|  technician|  single|secondary| 	no| 	29|	yes|  no|unknown|  5|  may| 	151|   	1|   -1|   	0| unknown|  2|[29.0,5.0,151.0,-...|
| 33|entrepreneur| married|secondary| 	no|  	2|	yes| yes|unknown|  5|  may|  	76|   	1|   -1|   	0| unknown|  2|[2.0,5.0,76.0,-1....|
| 47| blue-collar| married|  unknown| 	no|   1506|	yes|  no|unknown|  5|  may|  	92|   	1|   -1|   	0| unknown|  2|[1506.0,5.0,92.0,...|
| 33| 	unknown|  single|  unknown| 	no|  	1| 	no|  no|unknown|  5|  may| 	198|   	1|   -1|   	0| unknown|  2|[1.0,5.0,198.0,-1...|
| 35|  management| married| tertiary| 	no|	231|	yes|  no|unknown|  5|  may| 	139|   	1|   -1|   	0| unknown|  2|[231.0,5.0,139.0,...|
| 28|  management|  single| tertiary| 	no|	447|	yes| yes|unknown|  5|  may| 	217|   	1|   -1|   	0| unknown|  2|[447.0,5.0,217.0,...|
| 42|entrepreneur|divorced| tertiary|	yes|  	2|	yes|  no|unknown|  5|  may| 	380|   	1|   -1|   	0| unknown|  2|[2.0,5.0,380.0,-1...|
| 58| 	retired| married|  primary| 	no|	121|	yes|  no|unknown|  5|  may|  	50|   	1|   -1|   	0| unknown|  2|[121.0,5.0,50.0,-...|
| 43|  technician|  single|secondary| 	no|	593|	yes|  no|unknown|  5|  may|  	55|   	1|   -1|   	0| unknown|  2|[593.0,5.0,55.0,-...|
| 41|  	admin.|divorced|secondary| 	no|	270|	yes|  no|unknown|  5|  may| 	222|   	1|   -1|   	0| unknown|  2|[270.0,5.0,222.0,...|
| 29|  	admin.|  single|secondary| 	no|	390|	yes|  no|unknown|  5|  may| 	137|   	1|   -1|   	0| unknown|  2|[390.0,5.0,137.0,...|
| 53|  technician| married|secondary| 	no|  	6|	yes|  no|unknown|  5|  may| 	517|   	1|   -1|   	0| unknown|  2|[6.0,5.0,517.0,-1...|
| 58|  technician| married|  unknown| 	no| 	71|	yes|  no|unknown|  5|  may|  	71|   	1|   -1|   	0| unknown|  2|[71.0,5.0,71.0,-1...|
| 57|	services| married|secondary| 	no|	162|	yes|  no|unknown|  5|  may| 	174|   	1|   -1|   	0| unknown|  2|[162.0,5.0,174.0,...|
| 51| 	retired| married|  primary| 	no|	229|	yes|  no|unknown|  5|  may| 	353|   	1|   -1|   	0| unknown|  2|[229.0,5.0,353.0,...|
| 45|  	admin.|  single|  unknown| 	no| 	13|	yes|  no|unknown|  5|  may|  	98|   	1|   -1|   	0| unknown|  2|[13.0,5.0,98.0,-1...|
| 57| blue-collar| married|  primary| 	no| 	52|	yes|  no|unknown|  5|  may|  	38|   	1|   -1|   	0| unknown|  2|[52.0,5.0,38.0,-1...|
| 60| 	retired| married|  primary| 	no| 	60|	yes|  no|unknown|  5|  may| 	219|   	1|   -1|   	0| unknown|  2|[60.0,5.0,219.0,-...|
| 33|	services| married|secondary| 	no|  	0|	yes|  no|unknown|  5|  may|  	54|   	1|   -1|   	0| unknown|  2|[0.0,5.0,54.0,-1....|
+---+------------+--------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+--------------------+
only showing top 20 rows

scala> // We change the column y to the label column

scala> val cambio = fea.withColumnRenamed("y", "label")
cambio: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]

scala> val feat = cambio.select("label","features")
feat: org.apache.spark.sql.DataFrame = [label: int, features: vector]

scala> feat.show(1)
+-----+--------------------+
|label|        	features|
+-----+--------------------+
|	2|[2143.0,5.0,261.0...|
+-----+--------------------+
only showing top 1 row

scala> // Logistic Regression algorithm

scala> val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
logistic: org.apache.spark.ml.classification.LogisticRegression = logreg_f6bae136adc2

scala> // Fit of the model

scala> val logisticModel = logistic.fit(feat)
21/01/12 19:35:58 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/01/12 19:35:58 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
logisticModel: org.apache.spark.ml.classification.LogisticRegressionModel = LogisticRegressionModel: uid = logreg_f6bae136adc2, numClasses = 3, numFeatures = 5

scala> // Impression of coefficients and interception

scala> println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
org.apache.spark.SparkException: Multinomial models contain a matrix of coefficients, use coefficientMatrix instead.
  at org.apache.spark.ml.classification.LogisticRegressionModel.coefficients(LogisticRegression.scala:955)
  ... 52 elided

scala> val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
logisticMult: org.apache.spark.ml.classification.LogisticRegression = logreg_e3fb445de801

scala> val logisticMultModel = logisticMult.fit(feat)
logisticMultModel: org.apache.spark.ml.classification.LogisticRegressionModel = LogisticRegressionModel: uid = logreg_e3fb445de801, numClasses = 3, numFeatures = 5

scala> println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
Multinomial coefficients: 3 x 5 CSCMatrix

scala> println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")
Multinomial intercepts: [-7.827431229384973,2.903059293515478,4.924371935869495]

Corrida 1 Arbol de Decision
-------------------+-----+--------------------+                                
|         prediction|label|            features|
+-------------------+-----+--------------------+
|0.04275244299674267|    0|[-6847.0,21.0,206...|
|0.15129151291512916|    0|[-3372.0,29.0,386...|
| 0.6363636363636364|    0|[-2827.0,31.0,843...|
|0.02792220296553052|    0|[-2604.0,18.0,142...|
|0.02792220296553052|    0|[-2082.0,28.0,123...|
+-------------------+-----+--------------------+
rmse: Double = 0.283860295997569  
Root Mean Squared Error (RMSE) on test data = 0.283860295997569
DecisionTreeRegressionModel: uid=dtr_f25620ce31e6, depth=5, numNodes=63, numFeatures=5
  If (feature 2 <= 564.5)
   If (feature 2 <= 199.5)
    If (feature 3 <= -0.5)
     If (feature 2 <= 107.5)
      If (feature 2 <= 72.5)
       Predict: 0.0028453181583031556
      Else (feature 2 > 72.5)


       Predict: 0.015809051456912585
     Else (feature 2 > 107.5)
      If (feature 0 <= 1041.5)
       Predict: 0.02792220296553052
      Else (feature 0 > 1041.5)
       Predict: 0.06294034758102396
    Else (feature 3 > -0.5)
     If (feature 2 <= 144.5)
      If (feature 3 <= 164.5)
       Predict: 0.09724473257698542
      Else (feature 3 > 164.5)
       Predict: 0.027605244996549344
     Else (feature 2 > 144.5)
      If (feature 3 <= 164.5)
       Predict: 0.3586206896551724
      Else (feature 3 > 164.5)
       Predict: 0.11690363349131122
   Else (feature 2 > 199.5)
    If (feature 3 <= 3.5)
     If (feature 2 <= 375.5)
      If (feature 0 <= 218.5)
       Predict: 0.04275244299674267
      Else (feature 0 > 218.5)
       Predict: 0.09826302729528535
     Else (feature 2 > 375.5)
      If (feature 2 <= 479.5)
       Predict: 0.15129151291512916
      Else (feature 2 > 479.5)
       Predict: 0.21672555948174324
    Else (feature 3 > 3.5)
     If (feature 3 <= 191.5)
      If (feature 3 <= 95.5)
       Predict: 0.6079545454545454
      Else (feature 3 > 95.5)
       Predict: 0.4065244667503137
     Else (feature 3 > 191.5)
      If (feature 0 <= 1634.5)
       Predict: 0.15502183406113537
      Else (feature 0 > 1634.5)
       Predict: 0.3537117903930131
  Else (feature 2 > 564.5)
   If (feature 2 <= 897.5)
    If (feature 3 <= 8.5)
     If (feature 2 <= 677.0)
      If (feature 1 <= 13.5)
       Predict: 0.3698630136986301
      Else (feature 1 > 13.5)
       Predict: 0.2657200811359026
     Else (feature 2 > 677.0)
      If (feature 1 <= 29.5)
       Predict: 0.4250614250614251
      Else (feature 1 > 29.5)
       Predict: 0.6363636363636364
    Else (feature 3 > 8.5)
     If (feature 1 <= 21.5)
      If (feature 1 <= 15.5)
       Predict: 0.5534883720930233
      Else (feature 1 > 15.5)
       Predict: 0.32989690721649484
     Else (feature 1 > 21.5)
      If (feature 3 <= 283.5)
       Predict: 0.8653846153846154
      Else (feature 3 > 283.5)
       Predict: 0.2
   Else (feature 2 > 897.5)
    If (feature 1 <= 29.5)
     If (feature 0 <= 1.5)
      If (feature 3 <= 283.5)
       Predict: 0.4962962962962963
      Else (feature 3 > 283.5)
       Predict: 0.8571428571428571
     Else (feature 0 > 1.5)
      If (feature 0 <= 135.5)
       Predict: 0.7019230769230769
      Else (feature 0 > 135.5)
       Predict: 0.5779944289693594
    Else (feature 1 > 29.5)
     If (feature 0 <= 3686.5)
      If (feature 0 <= 0.5)
       Predict: 0.6666666666666666
      Else (feature 0 > 0.5)
       Predict: 0.8857142857142857
     Else (feature 0 > 3686.5)
      If (feature 0 <= 4995.0)
       Predict: 0.5
      Else (feature 0 > 4995.0)
       Predict: 0.6666666666666666
Used Memory: 153.08808 mb
Free Memory: 42.261744 mb
Total Memory: 214.63449599999998 mb
Max Memory: 1037.9591679999999 mb
Long = 1610080305866
durationSeconds: String = 887s


Corrida 1  Maquina de vectores
startTimeMillis: Long = 1610206030727
+----------+-----+--------------------+                                    	 
|prediction|label|        	features|
+----------+-----+--------------------+
|   	0.0|	0|[-6847.0,21.0,206...|
|   	0.0|	0|[-3372.0,29.0,386...|
|   	0.0|	0|[-3313.0,9.0,153....|
|   	0.0|	0|[-2712.0,2.0,253....|
|   	0.0|	0|[-2604.0,18.0,142...|
+----------+-----+--------------------+
accuracy: Double = 0.8844117647058823 
Test Error = 0.11558823529411766
Accuracy = 0.8844117647058823
Coefficients: [0.0,-0.008894855345256487,7.071595392375417E-4,1.4692542286840073E-4,0.010516073853206414] Intercept: -1.1035705126490727
mb: Double = 1.0E-6
Used Memory: 301.30192 mb
Free Memory: 376.852088 mb
Total Memory: 694.6816 mb
Max Memory: 954.728448 mb
endTimeMillis: Long = 1610206533175
durationSeconds: String = 502s








Conclusiones

Después de haber corrido los cuatro diferentes algoritmos  es bueno indicar que cada algoritmo escogido mostrará un comportamiento diferente, ya sea por la técnica utilizada para procesar y clasificar, como por la distribución y naturaleza de los datos suministrados. Así de cada algoritmo podremos extraer valores como la precisión que ha tenido con un conjunto de datos de entrenamiento o el consumo de recursos de la máquina en la que se ejecute. Lo que buscamos así para la comparativa a llevar a cabo en este trabajo es, por un lado entender de forma más precisa cómo funciona cada clasificador y, por otra parte, definiremos unos criterios que nos ayudarán a tomar la decisión de cuál es el mejor para el tratamiento del problema planteado. Los criterios que podemos tomar para nuestro análisis y comparativa pueden ser: 
Recursos del sistema ○ Memoria usada para ajuste o entrenamiento ○ Tiempo necesario para el ajuste ○ Tiempo necesario para la predicción de las etiquetas.
Valores del propio algoritmo ○ Exactitud, sensibilidad, precisión entre otros para clasificar un conjunto de test de datos sin etiqueta y otros valores que se pueden medir y que explicaremos posteriormente. 
El Perceptrón Multicapa: es una generalización del Perceptrón Simple y surgió como consecuencia de las limitaciones que tenía a la hora de clasificar conjuntos de datos que no eran linealmente separables.
Regresión logística: este algoritmo es muy simple de implementar pero sin embargo es difícil que converja si los datos que estemos tratando no son separables linealmente. Es un modelo de regresión donde la variable dependiente es categórica, aquella que puede coger valores fijos o un número de valores posibles para la resolución de problemas, debido a su capacidad como aproximador fundamental y a su facilidad de uso y aplicabilidad
Máquinas de vectores de soporte: puede ser considerado como una ampliación del Perceptrón. Es importante saber de qué forma este algoritmo obtiene el hiperplano más óptimo para una clasificación binaria, un hiperplano que se encargará de separar aquellos ejemplos que pertenecen a una u otra clase. 
Árbol de decisión: si hay algo que bien podemos destacar de este tipo de algoritmos y la visualización de su efectividad es por sus reglas de clasificación fácilmente comprensibles para los humanos. En los árboles de decisión se realiza un agrupamiento de los datos basándose en los valores de los atributos de los datos que tenemos. Se realiza una división de clases atendiendo a aquel atributo que mejor distinga entre unos otros y se aplica recursivamente este proceso hasta que todos los datos de un subconjunto que se esté tratando pertenezca a la misma clase son muy fáciles de interpretar debido a la lógica que utiliza a la hora de tomar las decisiones, las posibilidades de sus hiperparámetros son muy pocas, donde la interpretación y visualización son muy fáciles.

Referencias bibliográficas

Classification and regression - Spark 2.3.0 Documentation. (2012). Classification and regression. https://spark.apache.org/docs/2.3.0/ml-classification-regression.html.

Heras, J. M. (2019, 28 mayo). Máquinas de Vectores de Soporte (SVM).Artificial.net. https://www.iartificial.net/maquinas-de-vectores-de-soporte-svm/.

Castro, A. A. S. (2020, 19 mayo). Una introducción a los Árboles de Decisión. DABIA.https://www.grupodabia.com/post/2020-05-19-arbol-de-decision/#:%7E:text=Los%20%C3%A1rboles%20de%20decisi%C3%B3n%20son,de%20Machine%20learning%20m%C3%A1s%20utilizadas.&text=La%20estrategia%20para%20construir%20un,m%C3%A9todo%20heur%C3%ADstico%20llamado%20partici%C3%B3n%20recursiva.

Pérez, F. M. D. (2000, 1 diciembre). La regresión logística: una herramienta versátil | Nefrología. La regresión logística. https://www.revistanefrologia.com/es-la-regresion-logistica-una-herramienta-versatil-articulo-X0211699500035664.

Buitrago, B. (2020, 22 septiembre). Redes Neuronales — Perceptrón Multicapa I - iWannaBeDataDriven. Medium. https://medium.com/iwannabedatadriven/redes-neuronales-perceptr%C3%B3n-multicapa-i-d8c05a88857e.

Apache Spark. (2012). Spark SQL, DataFrames and Datasets Guide. 2020 14, de Apache Spark Sitio web: https://spark.apache.org/docs/latest/sql-programming-guide.html

Lenguajes de Programacion. (2015). Scala. 2020 Enero, de Web Sitio web: 
https://lenguajesdeprogramacion.net/scala/




Anexos 
Code Support Vector Machines

import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder,IndexToString}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession


// Minimiza los erorres mostrados 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Valores para medir el rendimiento
val runtime = Runtime.getRuntime
val startTimeMillis = System.currentTimeMillis()

// Se inicia una sesion en spark
val spark = SparkSession.builder().getOrCreate()

// Se cargan los datos en la variable "data" en el formato "libsvm"
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

// Se imprime el schema del dataFrame
data.printSchema()

//Convertimos los valores de la columna "y" en numerico
val change1 = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),0).otherwise(col("y")))
val newDF = change2.withColumn("y",'y.cast("Int"))


// Generamos un vector con los nombres de las columnas a evaluar
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))

// Se rellena el vector con los valores
val features = vectorFeatures.transform(newDF)

//Renombramos la columna "y" a Label
val featuresLabel = features.withColumnRenamed("y", "label")

//Indexamos las columnas label y features
val dataIndexed = featuresLabel.select("label","features")

// Indexamos las etiquetas
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed) // features with > 4 distinct values are treated as continuous.

// Dividimos los datos, 70% entrenamiento y 30% prueba
//val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))
val splits = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)

// Creamos el modelo y le indicamos las columnas a utilizar
val lsvc = new LinearSVC().setMaxIter(100).setRegParam(0.1)

val lsvcModel = lsvc.fit(test)

// Metemos todo en una tuberia
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lsvc))

// Ahora si entrenamos el modelo con el 70% de los datos
val model = pipeline.fit(trainingData)

// Realizamos la predicción de los datos con el 30% de la data
val predictions = model.transform(testData)

// imprimiemos los primero s5 registros
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


Code Decision Tree
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder,IndexToString}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

// Minimiza los erorres mostrados 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Valores para medir el rendimiento
val runtime = Runtime.getRuntime
val startTimeMillis = System.currentTimeMillis()

// Se inicia una sesion en spark
val spark = SparkSession.builder().getOrCreate()

// Se cargan los datos en la variable "data" en el formato "csv"
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

// Se imprime el schema del dataFrame
data.printSchema()

//Convertimos los valores de la columna "y" en numerico
val change1 = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),0).otherwise(col("y")))
val newDF = change2.withColumn("y",'y.cast("Int"))


// Generamos un vector con los nombres de las columnas a evaluar
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))

// Se rellena el vector con los valores
val features = vectorFeatures.transform(newDF)

//Renombramos la columna "y" a Label
val featuresLabel = features.withColumnRenamed("y", "label")

//Indexamos las columnas label y features
val dataIndexed = featuresLabel.select("label","features")

// Indexamos las etiquetas
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed) 

// Dividimos los datos, 70% entrenamiento y 30% prueba
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))

// Creamos el modelo y le indicamos las columnas a utilizar
val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")

// Metemos todo en una tuberia
val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))

// Ahora si entrenamos el modelo con el 70% de los datos
val model = pipeline.fit(trainingData)

// Realizamos la predicción de los datos con el 30% de la data
val predictions = model.transform(testData)

// imprimiemos los primero s5 registros
predictions.select("prediction", "label", "features").show(5)

// Seleecionamos las columnas y el valor del error
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

Code Logistic Regression
/Importamos las librerias que utilizaremos para realizar nuestro ejercicio
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

// Minimizamos los errores 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Creamos una sesion de spark 
val spark = SparkSession.builder().getOrCreate()

//cargamos nuestro archivo CSV

val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

// Se imprime el schema
df.printSchema()

// Visualizamos nuestro Dataframe 
df.show()


//Modificamos la columna de strings a datos numericos 
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
//Desplegamos la nueva columna
newcolumn.show()

//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
//Mostramos la nueva columna
fea.show()
//Cambiamos la columna y a la columna label
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
feat.show(1)
//Algoritmo Logistic Regression
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
// Fit del modelo
val logisticModel = logistic.fit(feat)
//Impresion de los coegicientes y de la intercepcion
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")

Code Multilayer perceptrón


//Importamos las librerias que utilizaremos para realizar nuestro ejercicio
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

// Minimizamos los errores 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


//Creamos una sesion de spark 
val spark = SparkSession.builder().getOrCreate()

//cargamos nuestro archivo CSV

val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

// Se imprime el schema
df.printSchema()

// Visualizamos nuestro Dataframe 
df.show()




//Modificamos la columna de strings a datos numericos 
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))

//Generamos la tabla features
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)

//Cambiamos la columna a label 
val cambio = fea.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")

//Multilayer perceptron
//Dividimos los datos en un arreglo en partes de 70% y 30%
val split = feat.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = split(0)
val test = split(1)

// Especificamos las capas para la red neuronal
val layers = Array[Int](5, 2, 2, 4)


//Creamos el entrenador con sus parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

//Entrenamos el modelo
val model = trainer.fit(train)

//Imprimimos la exactitud
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")





     
</br>

