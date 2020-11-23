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
   	 
    	1.-Create a list called "list" with the elements "red", "white", "black"
    	2.-Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
    	3.-Bring the "list" "green", "yellow", "blue" items
    	4.-Create a number array in the 1-1000 range in 5-in-5 steps
    	5.-What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets
    	6.-Create a mutable map called names containing the following"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
    	6.-a. Print all map keys
    	7.-b. Add the following value to the map ("Miguel", 23)
   	 
#### In this practice  we created lists with different colors, aslo we added elements to the lists, and finally  we created an array with ranges and that they count from 5 to 5. We made a mutable map and printed.
   	 
#### &nbsp;&nbsp;&nbsp;&nbsp; Code.
```scala
     	/*1.-Create a list called "list" with the elements "red", "white", "black"*/
     	var lista = collection.mutable.MutableList("rojo","blanco","negro") 	 

     	/*2.-Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"*/
      	lista += ("verde","amarillo", "azul", "naranja", "perla")


     	/*3.-Bring the "list" "green", "yellow", "blue" items*/
         	lista(3)
         	lista(4)
         	lista(5)

     	/*4.-Create a number array in the 1-1000 range in 5-in-5 steps*/
           	var v = Range(1,1000,5)


     	/*5.-What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets*/
          	var l = List(1,3,3,4,6,7,3,7)
           	l.toSet


     	/*6.-Create a mutable map called names containing the following"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27*/
      	var map=collection.mutable.Map(("Jose", 20),("Luis", 24),("Ana", 23),("Susana", "27"))

     	/*6.-a. Print all map keys*/
       	map.keys

     	/*7.-b. Add the following value to the map ("Miguel", 23)*/
      	map += ("Miguel"->23)

```