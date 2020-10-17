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
