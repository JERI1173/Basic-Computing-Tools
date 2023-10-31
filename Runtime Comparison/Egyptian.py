"""
Egyptian algorithm
"""
import numpy as np

def egyptian_multiplication(a, n):
    """
    returns the product a * n

    assume n is a nonegative integer
    """
    def isodd(n):
        """
        returns True if n is odd
        """
        return n & 0x1 == 1

    if n == 1:
        return a
    if n == 0:
        return 0

    if isodd(n):
        return egyptian_multiplication(a + a, n // 2) + a
    else:
        return egyptian_multiplication(a + a, n // 2)


if __name__ == '__main__':
    # this code runs when executed as a script
    for a in [1,2,3]:
        for n in [1,2,5,10]:
            print("{} * {} = {}".format(a, n, egyptian_multiplication(a,n)))


def power(a, n):
    """
    computes the power a ** n

    assume n is a nonegative integer
    """
    result = 1 # initiate variable "result"
    
    if n == 0: 
        return 1
    else:             
        while n > 0:
            if n % 2 != 0: # n is odd
                result = result * a
                n = n - 1
            else: # n is even
                a = a * a
                n = n // 2
        return result  
    pass

def EgyptianAlgo_matrix(A, n):
    I = np.array([[1,0], # initaite an numpy array
                  [0,1]])
    result = I
    
    while n > 0: # Egyptian comes in with the same structure
        if n % 2 != 0:# n is odd
            result = np.dot(result, A) # result multiplied by A
            n = n - 1  
        else: # n is even
            A = np.dot(A, A) # A multiplied by A
            n = n // 2
    
    return result