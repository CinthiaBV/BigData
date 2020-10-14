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

