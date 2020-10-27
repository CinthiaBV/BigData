# Unit_1

### <p align="center" > TECNOLÓGICO NACIONAL DE MÉXICO INSTITUTO TECNOLÓGICO DE TIJUANA SUBDIRECCIÓN ACADÉMICA DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN PERIODO: Agosto-Diciembre  2020</p>

###  <p align="center">  Carrera: Ing. En Sistemas Computacionales. 
### <p align="center"> Materia: 	Datos Masivos (BDD-1704 SC9A	).</p>

### <p align="center">  Maestro: Jose Christian Romero Hernandez	</p>
### <p align="center">  No. de control y nombre del alumno: 15211916 - Vargas Garcia Cinthia Gabriela</p>

## Index
&nbsp;&nbsp;&nbsp;[Practice 1](#practice-1)  
&nbsp;&nbsp;&nbsp;[Practice 2](#practice-2)

&nbsp;&nbsp;&nbsp;[Practice 3](#practice-3)    
&nbsp;&nbsp;&nbsp;[Practice 4](#practice-4) 

&nbsp;&nbsp;&nbsp;[Practice 5](#practice-5) 

&nbsp;&nbsp;&nbsp;[Practice GitHub](#practice-GitHub) 

&nbsp;&nbsp;&nbsp;[Homework 1](#Homewok-1) 

&nbsp;&nbsp;&nbsp;[Investigation 1 ](#Investigation-1) 

&nbsp;&nbsp;&nbsp;[Test 1 ](#Test-1)  

 


### &nbsp;&nbsp;Practice 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
   
        1.-Develop an algorithm in scala that calculates the radius of a circle
        2.-Develop an algorithm in scala that tells me if a number is a cousin
        3.-Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet"
        4.-Given the variable message = "Hi Luke, I'm your father!" use slilce to extract the  sequence "Luke"
        5.-What is the difference in value and a variable in scala?
        6.-Given the tuple ((2,4,5), (1,2,3), (3,1416,23))) the number 3,1416 returns
        
        
#### In this practice, we did  the connotation of marios methods, mathematical in order to get the radius of a circle.
     
</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala   
      /*1.Develop an algorithm in scala that calculates the radius of a circle*/
            var a=3
     var r=math.sqrt(a/math.Pi)        
```        
```scala     
      /*2. Develop an algorithm in scala that tells me if a number is a cousin*/
        var t = ((2,4,5),(1,2,3),(3.1416,23))
        t._3._1
``` 

```scala  
      /*3. Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet"n*/
        var bird="tweet"
        printf(s"Estoy ecribiendo un %s",bird)
``` 

```scala   
        /*4. Given the variable message = "Hi Luke, I'm your father!" use slilce to extract the  sequence "Luke"*/
        var mensaje = "Hola Luke yo soy tu padre!"
        mensaje.slice(5,9)
``` 

```scala  
      /*5. What is the difference in value and a variable in scala?*/
       Value (val) is immutable once assigned the value this cannot be changed
       Variable (var) once assigned you can reassign the value, as long as the new value sea of the same type
``` 

```scala  
      /*6. Given the tuple ((2,4,5), (1,2,3), (3,116,23))) the number 3,141 returns*/
       var t = ((2,4,5),(1,2,3),(3.1416,23))
       t._3._1
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



#### In this practice, we created array, list and map.
     
</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala  
/*1.-Create a list called "list" with the elements "red", "white", "black"*/
         var lista = collection.mutable.MutableList("rojo","blanco","negro") 

``` 
```scala
         /*2.-Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"*/
          lista += ("verde","amarillo", "azul", "naranja", "perla")

``` 

 ```scala
         /*3.-Bring the "list" "green", "yellow", "blue" items*/
             lista(3)
             lista(4)
             lista(5)
``` 

            
 ```scala
         /*4.-Create a number array in the 1-1000 range in 5-in-5 steps*/
               var v = Range(1,1000,5)
```                

 ```scala
         /*5.-What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets*/
              var l = List(1,3,3,4,6,7,3,7)
               l.toSet
```                

 ```scala
         /*6.-Create a mutable map called names containing the following"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27*/
          var map=collection.mutable.Map(("Jose", 20),("Luis", 24),("Ana", 23),("Susana", "27"))
```           

```scala
         /*6.-a. Print all map keys*/
           map.keys
```            

```scala
         /*7.-b. Add the following value to the map ("Miguel", 23)*/
          map += ("Miguel"->23)
```           

### &nbsp;&nbsp;Practice 3.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

1.- Analysis

#### In this practice, we analyse code lines.
     
</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala

//Define list event
	//cicle
	// if n is a number divisible on 2 print is even
	// else n is not a number divisible on 2  print is odd
	def listEvens(list:List[Int]): String ={
	    for(n <- list){
	        if(n%2==0){
	            println(s"$n is even")
	        }else{
	            println(s"$n is odd")
	        }
	    }
	    return "Done"
	}
	// In this section of the code we declare 2 lists
	// and 2 Lis Events
	//Based on the elements of this list 
	//returning messages of is even or is odd based on the list elements
	val l = List(1,2,3,4,5,6,7,8)
	val l2 = List(4,3,22,55,7,8)
	listEvens(l)
	listEvens(l2)
	

	//3 7 afortunado
	//List event
	def afortunado(list:List[Int]): Int={
	    var res=0
	    for(n <- list){
	        if(n==7){
	            res = res + 14
	        }else{
	            res = res + n
	        }
	    }
	    return res
	}
	//Values of afortunado
	//Based of values of this list we gona to receive results based on the condictional of for cicle
	val af= List(1,7,7)//
	//Print the result afortunado 29
	//1 = 1    7=14      7=14
	//1+14+14 = 29
	println(afortunado(af))
	//Define event list Balance
	def balance(list:List[Int]): Boolean={
	    var primera = 0
	    var segunda = 0
	

	    segunda = list.sum
	    //Primera Sum values to our list
	    //Segunda rest values to our list
	

	    for(i <- Range(0,list.length)){
	        primera = primera + list(i)
	        segunda = segunda - list(i)
	

	        if(primera == segunda){
	            return true
	        }
	    }
	    return false 
	}
	//List ELements
	val bl = List(3,2,1)
	val bl2 = List(2,3,3,2)
	val bl3 = List(10,30,90)
	//run
	balance(bl)
	balance(bl2)
	balance(bl3)
	//Define palindromo
	//Reverse Palabra(Word)
	// if word dont are exactly the same word reverse
	//return false
	def palindromo(palabra:String):Boolean ={
	    return (palabra == palabra.reverse)
	}
	

	val palabra = "OSO" 
	val palabra2 = "ANNA" 
	val palabra3 = "JUAN" 
	//Return Boolean
	//run
	println(palindromo(palabra))
	println(palindromo(palabra2))
	println(palindromo(palabra3))

```   


### &nbsp;&nbsp;Practice 4.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

        1.- Recursive version descending.
        2.- Explicit formula.
        3.- Iterative version.
        4.- Iterative version with variables.
        5.- Iterative vector version.
       
#### In this practice we did The Fibonacci Sequence.

</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

       
```scala
      /* Each of these calls, if it is greater than 1, will again be calling your previous    two.*/
    /* 1.- Recursive  descending.*/
    def fib1( n : Int) : Int =
    {
        if(n<2) n
        else fib1( n-1 ) + fib1( n-2 )
    }
```
```scala
       /*Fibonacci sequence uch that each number is the sum of the two preceding ones, starting from 0 and 1. Fibonacci numbers have the generating function*/
    /* 2.- Explicit formula.*/

    def fib2( n : Int ) : Int = {
        if(n<2)n
        else
        {
            var y = ((1+math.sqrt(5))/2)
            var k = ((math.pow(y,n)-math.pow((1-y),n))/math.sqrt(5))      
            k.toInt
        }
}

```
```scala
/* 5.- Iterative vector version.*/
    def fib5(n:Int): Int={
    if(n<2)n    
    else{        
        var b = List.range(0,n+1)
        for(k<-Range(2,n+1)){            
            b = b.updated(k,(b(k-1)+b(k-2)))            
        }            
        b(n)
     }     
}

```

### &nbsp;&nbsp;Practice 5.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

1.-Examples of 15 Data Frames 

#### In this practice we did examples of DataFrames.

</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala

//Import Data frames
import spark.implicits._

val Information = Seq (
    ("Cinthia","Sales","B.C", 45,2,2333),
    ("Laura","HR","B.C", 28,21,3333),
    ("Karla","QC","B.C", 45,20,6333),
    ("Hugo","QC","B.C", 35,4,6333),
    ("Anabel","Sales","B.C", 21,54,7899)
)
val df= simpleData.toDF("name","area","city","age","hours","payment")

//1 Show the information of Dataframe
df.show()

//2 Returns the population covariance for two columns.
df.select(covar_pop("Sales","Sales")).show()

//3 Returns the sample covariance for two columns.

df.select(covar_samp("Sales","Sales2")).show()

//4 Returns the first element in a column.
df.select(first("Sales")).show()


//5 Select only the last item of Sales

df.select(last("Sales")).show()

//6 Returns the sum of all distinct values in a column.
df.select(sumDistinct("Sales")).show()

// 7 Returns the population covariance for two columns.

df.select(covar_pop("Sales")).show()

//8 Function returns the average of values in the input column.
df.select(avg("Sales")).show()


// 9 Returns the unbiased variance of the values in a column.
 
df.select(var_samp("Sales")).show()

//10 Returns the count of distinct items in a group.
df.select(approx_count_distinct("Sales")).show()


//11 Returns all values from an input column with duplicates.
df.select(collect_list("Sales")).show()
 

// 12 Returns the kurtosis of the values in a group. 
df.select(kurtosis("Sales")).show()

//13 Returns the skewness of the values in a group. 
df.select(skewness("Sales")).show()


//14 Returns the maximum value in a column.
df.select(max("Company")).show()

// 15 Returns the Pearson Correlation Coefficient for two columns.
df.select(corr("Sales","Sales2")).show()

```

### &nbsp;&nbsp;Practice GitHub.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

1.-Git Flow Practice 

#### In this practice we did examples classes.

</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala

Introduction

Git is an open source version specific control system created by Linus Torvalds in 2005.

Specifically, Git is a distributed version control system, which means that the entire code base and its history are 
available on every developer's computer, allowing easy access to branching and merging.

GitHub is a non-profit company that offers a hosting service for repositories stored in the cloud. Essentially, it makes it easier
 for individuals and teams to use Git as the control and collaboration version. The GitHub interface is fairly easy to use for the novice 
 developer who wants to take advantage of Git. Without GitHub, using Git generally requires a bit more tech savvy and command line usage.

```

### &nbsp;&nbsp;Homework 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

1.-Pearson

#### In this practice we did investigation of Pearson.

</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala
Content

What is Pearson's correlation coefficient? 3
How the Pearson 3 Correlation Coefficient is Calculated
Interpretation of the Karl Pearson correlation coefficient 3
Advantages and disadvantages of the Pearson 4 correlation coefficient
Bibliography 5




What is Pearson's correlation coefficient?

Pearson's correlation coefficient is a test that measures the statistical relationship between two continuous variables. 
If the association between the elements is not linear, then the coefficient is not adequately represented.
The correlation coefficient can take a range of values ​​from +1 to -1. A value of 0 indicates that there is no association 
between the two variables. A value greater than 0 indicates a positive association. That is, as the value of one variable 
increases, so does the value of the other. A value less than 0 indicates a negative association; that is, as the value of 
one variable increases, the value of the other decreases.
To carry out the Pearson correlation it is necessary to fulfill the following:
• The measurement scale must be an interval or ratio scale.
• The variables must be approximately distributed.
• The association must be linear.
• There should be no outliers in the data.

```

### &nbsp;&nbsp;Investigation 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

1.-Pair Coding

#### In this practice we did investigation of Pair Coding.

</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala
Content

Pair Programming 3
What is pair programming? 3
Internship in pair programming 3
Advantages and disadvantages of pair programming 4
What are the benefits of this approach? 5



Pair Programming

Developing new software is not an easy process. Depending on the size of the program, a large number of 
potential junctures, roles, and problematic issues will need to be considered. Even the most expert software 
developers can become disoriented. Hence, in recent years other more modern working methods have been developed 
that allow more efficient programming and generate error-free code: Scrum and Kanban serve, for example, to improve
 the complete system.
Pair programming doesn't try to be all that encompassing: developers always work in pairs on code. How does it 
work and what are the advantages of this working method?
What is pair programming?

The method known as pair programming (in Spanish, pair programming) is used mainly in agile software development and,
 more specifically, in extreme programming (XP). Pair programming specifies that there are always two people working 
 on the code at the same time, and that they sit together as much as possible. One is in charge of writing the code
  and the other of supervising it in real time. At the same time, they are constantly exchanging impressions: they 
  discuss problems, find solutions and develop creative ideas.
  Typically, these two workers are assigned different roles: the programmer who has been assigned the pilot role is 
  responsible for writing the code. The programmer who has been assigned the copilot role supervises that code. One of 
  the rules of pair programming states that these two roles are exchanged regularly (at short intervals). In this way, 
  a possible hierarchical gap is avoided: equality between both workers is promoted and a fluid exchange of roles is achieved.
In addition, ideally, the workspace is also tailored to the specific requirements of pair programming. Each worker must have 
their own mouse, keyboard and screen, which will always show the same information as the colleague's.
Somewhat less common is the method called remote pair programming. In this case, the programmers do not sit together, but are 
located in completely different places. For this method to work, you must have special technical solutions. Even despite the distance,
 colleagues must have a direct line of communication and must be able to access the code and view modifications in real time.

 ```


 ### &nbsp;&nbsp;Test 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

#### &nbsp;&nbsp;&nbsp;&nbsp; Link del video https://youtu.be/-Zsap6dyRUo.


1. Start a simple Spark session.
2. Upload Netflix Stock CSV file, have Spark infer data types. 3. What are the names of the columns?
4. What is the scheme like?
5. Print the first 5 columns.
6. Use describe () to learn about the DataFrame.
7. Create a new dataframe with a new column called “HV Ratio” which is the relationship between the price in the “High” column versus the “Volume” column of shares traded for a day. (Hint: It is a column operation).
8. What day had the highest peak in the “Close” column?
9. Write in your own words in a comment of your code. What is the meaning of the Close column "Close"?
10. What is the maximum and minimum of the “Volume” column?
11.With Scala / Spark $ Syntax answer the following:
◦ Hint: Basically very similar to the dates session, you will have to create another dataframe to answer some of the items.
to. 
a. How many days was the “Close” column less than $ 600?
b. What percentage of the time was the “High” column greater than $ 500? 
c. What is the Pearson correlation between column "High" and column "Volume"? d. What is the maximum in the “High” column per year?
e. What is the “Close” column average for each calendar month?

Evaluation instructions

- Delivery time 4 days
- At the end put the code and the documentation with its explanation in the corresponding branch of your github, likewise make your explanation of the solution in your google drive in google document (Cover, Introduction, Development, etc). - Finally defend your development in a video of 8-10 min explaining your solution and comments, this will serve to give your rating of this evaluation practice, this video must be uploaded to YouTube to be shared by a public link (Use google meet with the cameras turned on and record your defense to build the video).


#### In this practice we did e Test 1.

</br>

#### &nbsp;&nbsp;&nbsp;&nbsp; Code.

```scala

//1. Start a simple Spark session.

import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()


//2. Upload Netflix Stock CSV file, have Spark infer data types. 

val df =spark.read.option("header","true").option("inferSchema","true").csv("Netflix_2011_2016.csv")

//3. What are the names of the columns?

df.columns
//4. What is the scheme like?

df.printSchema()
//5. Print the first 5 columns.

df.head(5)
//6. Use describe () to learn about the DataFrame.

df.describe().show()

//7.Create a of dataframe with column called "HV Ratio" which is the
//relationship between the price of the column "High" versus the column "Volume" of
//shares traded for one day.
val df2 = df.withColumn("HV Ratio",df("High")/df("Volume"))

//8.What day had the highest peak in the “Close” column?
df.orderBy($"High".desc).show(1)

//9.for me column close represents the average of the document
df.select(mean("Close")).show()

//10. What is the max and min of the Volume column?
df.select(max("Volume")).show()
df.select(min("Volume")).show()

// For Scala/Spark $ Syntax
import spark.implicits._

//11a. How many days was the Close lower than $ 600?
df.filter($"Close"<600).count()

//11.b What percentage of the time was the High greater than $500 ?
(df.filter($"High">500).count()*1.0/df.count())*100

//11.c What is the Pearson correlation between High and Volume?
df.select(corr("High","Volume")).show()

//11.d What is the max High per year?
val yeardf = df.withColumn("Year",year(df("Date")))
val yearmaxs = yeardf.select($"Year",$"High").groupBy("Year").max()
yearmaxs.select($"Year",$"max(High)").show()

//11.e What is the average Close for each Calender Month?
val monthdf = df.withColumn("Month",month(df("Date")))
val monthavgs = monthdf.select($"Month",$"Close").groupBy("Month").mean()
monthavgs.select($"Month",$"avg(Close)").show()

 ```