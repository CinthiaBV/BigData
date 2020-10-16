# Unit_1

### <p align="center" > TECNOLÓGICO NACIONAL DE MÉXICO INSTITUTO TECNOLÓGICO DE TIJUANA SUBDIRECCIÓN ACADÉMICA DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN PERIODO: Agosto-Diciembre  2020</p>

###  <p align="center">  Carrera: Ing. En Sistemas Computacionales. 
### <p align="center"> Materia: 	Datos Masivos (BDD-1704 SC9A	).</p>

### <p align="center">  Maestro: Jose Christian Romero Hernandez	</p>
### <p align="center">  No. de control y nombre del alumno: 15211916 - Vargas Garcia Cinthia Gabriela</p>

## Index
&nbsp;&nbsp;&nbsp;[Practice 1](#practice-1)  




### &nbsp;&nbsp;Practice 1.

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.
   
        1.-Develop an algorithm in scala that calculates the radius of a circle
        2.-Develop an algorithm in scala that tells me if a number is a cousin
        3.-Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet"
        4.-Given the variable message = "Hi Luke, I'm your father!" use slilce to extract the  sequence "Luke"
        5.-What is the difference in value and a variable in scala?
        6.-Given the tuple ((2,4,5), (1,2,3), (3,116,23))) the number 3,141 returns
        
        
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
&nbsp;&nbsp;&nbsp;[Practice 2](#practice-2)  

#### &nbsp;&nbsp;&nbsp;&nbsp; Instructions.

    1.-Create a list called "list" with the elements "red", "white", "black"
    2.-Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
    3.-Bring the "list" "green", "yellow", "blue" items
    4.-Create a number array in the 1-1000 range in 5-in-5 steps
    5.-What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets
    6.-Create a mutable map called names containing the following"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
    6.-a. Print all map keys
    7.-b. Add the following value to the map ("Miguel", 23)



#### In this practice, we created array, list and map
     
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
