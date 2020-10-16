  
//Recursive version descending.
/* Each of these calls, if it is greater than 1, will again be calling your previous    two.*/
    /* 1.- Recursive  descending.*/
def fib1( n : Int) : Int =
{
    if(n<2) n
    else fib1( n-1 ) + fib1( n-2 )
}

fib1(5)