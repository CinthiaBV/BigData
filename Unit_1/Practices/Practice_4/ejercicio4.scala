//Exercise 4
//Define fib4
//Declare 2 variables type int
def fib4(n:Int):Int = {
   var a:Int =0
    var b:Int =1
    // use for cicle
    for (k<- Range(0,n)){
    
        b = b+a
        a = b-a
        
    }
     //Return a   
    return a
    }

//result
fib4(9)
