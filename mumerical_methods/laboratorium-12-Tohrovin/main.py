import numpy as np
from typing import Union, Callable


def solve_euler(fun: Callable, t_span: np.array, y0: np.array):
    ''' 
    Funkcja umożliwiająca rozwiązanie układu równań różniczkowych z wykorzystaniem metody Eulera w przód.
    
    Parameters:
    fun: Prawa strona równania. Podana funkcja musi mieć postać fun(t, y). 
    Tutaj t jest skalarem i istnieją dwie opcje dla ndarray y: Może mieć kształt (n,); wtedy fun musi zwrócić array_like z kształtem (n,). 
    Alternatywnie może mieć kształt (n, k); wtedy fun musi zwrócić tablicę typu array_like z kształtem (n, k), tj. każda kolumna odpowiada jednej kolumnie w y. 
    t_span: wektor czasu dla którego ma zostać rozwiązane równanie
    y0: warunke początkowy równanai o wymiarze (n,)
    Results:
    (np.array): macierz o wymiarze (n,m) zawierająca w wkolumnach kolejne rozwiązania fun w czasie t_span.  

    '''
    y = np.zeros((len(t_span), np.size(y0)))
    if not isinstance(y0, int):

        y[0] = y0

        for n in range(len(t_span) - 1):
            y[n + 1] = y[n] + fun(t_span[n], y[n]) * (t_span[n + 1] - t_span[n])
    else:

        y[0] = y0 

        for n in range(len(t_span) - 1):
            y[n + 1] = y[n] + fun(t_span[n], y[n]) * (t_span[n + 1] - t_span[n])

        

    return y

