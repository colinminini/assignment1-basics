
from Class import Model as hhh

Model = hhh

def f(L: list[int]) -> int:
    '''
    Returns the maximum value inside L
    '''
    max = - 1

    for x in L:
        if x > max:
            max = x

    return max

if __name__ == '__main__':
    print(f([1,2,3,5,4]))
    Model.call()