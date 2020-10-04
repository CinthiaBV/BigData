/*1.Develop an algorithm in scala that calculates the radius of a circle*/
var a=3
var r=math.sqrt(a/math.Pi)        
/*2. Develop an algorithm in scala that tells me if a number is a cousin*/
var t = ((2,4,5),(1,2,3),(3.1416,23))
t._3._1
/*3. Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet"n*/
var bird="tweet"
printf(s"Estoy ecribiendo un %s",bird)
/*4. Given the variable message = "Hi Luke, I'm your father!" use slilce to extract the  sequence "Luke"*/
var mensaje = "Hola Luke yo soy tu padre!"
mensaje.slice(5,9)
/*5. What is the difference in value and a variable in scala?*/
Value (val) is immutable once assigned the value this cannot be changed
Variable (var) once assigned you can reassign the value, as long as the new value
sea of the same type
/*6. Given the tuple ((2,4,5), (1,2,3), (3,116,23))) the number 3,141 returns*/
var t = ((2,4,5),(1,2,3),(3.1416,23))
t._3._1