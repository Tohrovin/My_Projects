##
import numpy as np
from numpy.core.fromnumeric import size
import scipy
import pickle
import matplotlib.pyplot as plt

from typing import Union, List, Tuple


def chebyshev_nodes(n: int = 10) -> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)
    
    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if not isinstance(n, int):
        return None

    result = []

    for k in range(n + 1):
        result.append(np.cos(np.pi * k / n))

    return np.array(result)


def bar_czeb_weights(n: int = 10) -> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)
    
    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if not isinstance(n, int):
        return None

    result = []
    for k in range(n + 1):

        delta = 1
        if k == 0 or k == n:
            delta = 0.5

        if k % 2 == 0:
            weight = delta
        else:
            weight = -delta

        result.append(weight)

    return np.array(result)


def barycentric_inte(xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n. 
    
    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0 
     
    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if all(isinstance(i, np.ndarray) for i in [xi, yi, wi, x]):
        if xi.shape == yi.shape and yi.shape == wi.shape:
            Y = []
            for el in np.nditer(x):
                if el in xi:
                    # omijamy dzielenie przez 0
                    Y.append(yi[np.where(xi == el)[0][0]])
                else:
                    # wzór w drugiej formie
                    L = wi / (el - xi)
                    Y.append(yi @ L / sum(L))
            return np.array(Y)
    else:
        return None


def L_inf(xr: Union[int, float, List, np.ndarray], x: Union[int, float, List, np.ndarray]) -> float:
    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(xr, (int, float)) and isinstance(x, (int, float)):
        return np.abs(xr - x)

    elif isinstance(xr, List) and isinstance(x, List) :
        if len(xr) == len(x):
            sub = xr[:]
            for i in range(len(sub)):
                sub[i] = abs(xr[i] - x[i])
            return max(sub)
        else:
            return np.NaN

    elif isinstance(xr, np.ndarray) and isinstance(x, np.ndarray):
        if xr.shape == x.shape:
            return max(np.abs(xr - x))
        else:
            return np.NaN

    elif isinstance(xr, np.ndarray) and isinstance(x, List):
        x = np.array(x)
        if xr.shape == x.shape:
            return max(np.abs(xr - x))
        else:
            return np.NaN

    elif isinstance(xr, List) and isinstance(x, np.ndarray):
        xr = np.array(xr)
        if xr.shape == x.shape:
            return max(np.abs(xr - x))
        else:
            return np.NaN

    # if isinstance(xr, (int, float)) and isinstance(x, (int, float)):
    #     return abs(xr - x)
    #
    # if isinstance(xr, List) and isinstance(x, List):
    #     xr = np.array(xr)
    #     x = np.array(x)
    #
    # if isinstance(xr, np.ndarray) and isinstance(x, np.ndarray):
    #     if xr.size != x.size:
    #         return np.NaN
    #     else:
    #         return max(abs(xr - x))
    #
    # return np.NaN

    return np.NaN



