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