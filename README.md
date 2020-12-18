# Unit_3

### <p align="center" > TECNOLÓGICO NACIONAL DE MÉXICO INSTITUTO TECNOLÓGICO DE TIJUANA SUBDIRECCIÓN ACADÉMICA DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN PERIODO: Agosto-Diciembre  2020</p>

###  <p align="center">  Carrera: Ing. En Sistemas Computacionales. 
### <p align="center"> Materia: 	Datos Masivos (BDD-1704 SC9A	).</p>

### <p align="center">  Maestro: Jose Christian Romero Hernandez	</p>
### <p align="center">  No. de control y nombre del alumno: 15211916 - Vargas Garcia Cinthia Gabriela</p>
### <p align="center">  No. de control y nombre del alumno: 16210561 - Oliver Cardenas Jesus Alejandro</p>

### Index

&nbsp;&nbsp;&nbsp;[Test_3](#Test-3)    

&nbsp;&nbsp;&nbsp;[Link_Video](#Link-video)     

### &nbsp;&nbsp;Test 3.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
Develop the following instructions in Spark with the Scala programming language.

Objective:
The goal of this hands-on test is to try to group customers from specific regions of a wholesaler. This based on the sales of some product categories.

The data sources are in the repository:
https://github.com/jcromerohdz/BigData/blob/master/Spark_clustering/Wholesale%20customers%20data.csv

    1. Import a simple Spark session.
    2. Use lines of code to minimize errors
    3. Create an instance of the Spark session
    4. Import the Kmeans library for the clustering algorithm.
    5. Load the Wholesale Customers Data dataset
    6. Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data
    7. Import Vector Assembler and Vector
    8. Create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels
    9. Use the assembler object to transform feature_data
    10. Create a Kmeans model with K = 3
    11. Evaluate the groups using Within Set Sum of Squared Errors WSSSE and print the centroids.

Evaluation instructions
- Delivery time 4 days



 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala  
// Import a simple Spark session.
// Import Vector Assembler and Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors


// Use the lines of code to minimize errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Create an instance of the Spark session
val spark = SparkSession.builder().getOrCreate()

// Import the Kmeans library for the clustering algorithm.
import org.apache.spark.ml.clustering.KMeans

// We load the Wholesale Customers Data dataset
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Wholesale customers data.csv")

// The dataFrame is printed
data.printSchema()

// Select the following columns: Fres, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
// The dataframe is cleaned up the empty fields
val f_data= (data.select($"Fresh", $"Milk",$"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen"))

// and call this set feature_data
val f_data_clean = f_data.na.drop()


// Create a new Vector Assembler object for the feature columns
// as an input set, remembering there are no labels
val f_Data = (new VectorAssembler().setInputCols(Array("Fresh","Milk", "Grocery","Frozen", "Detergents_Paper","Delicassen")).setOutputCol("features"))

// Use the assembler object to transform feature_data
val features = f_Data.transform(f_data_clean)

// The Kmeans model is executed with k = 3
val kmeans = new KMeans().setK(3).setSeed(1L).setPredictionCol("cluster")
val model = kmeans.fit(features)


// We evaluate the groups using Within Set Sum of Squared Errors WSSSE and
val WSSE = model.computeCost(features)
println(s"Within set sum of Squared Errors = $WSSE")

// We print the clusters
println("Cluster Centers: ")
model.clusterCenters.foreach(println)




```

### &nbsp;&nbsp;Link_Video.

<<<<<<< HEAD
#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
1.- https://youtu.be/escA1Oyz0zM
=======
// We evaluate the groups using Within Set Sum of Squared Errors WSSSE the objective is to minimize the sum of squares of the distance between the points of each set: the Euclidean distance squared. This is the goal of WCSS
val WSSE = model.computeCost(features)
println(s"Within set sum of Squared Errors = $WSSE")

// We print the clusters
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

```
>>>>>>> aac5f0f99616290d6a571ece05524eac361623a0
