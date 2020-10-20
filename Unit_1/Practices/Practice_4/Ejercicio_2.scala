 Fibonacci sequence uch that each number is the sum of the two preceding ones, starting from 0 and 1. Fibonacci numbers have the generating function
    /* 2.- Version with explicit formula.*/
    def fib2( n : Int ) : Int = {
        if(n<2)n
        else
        {
            var y = ((1+math.sqrt(5))/2)
            var k = ((math.pow(y,n)-math.pow((1-y),n))/math.sqrt(5))      
            k.toInt
        }
}