import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[
    int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """

    if not isinstance(v, (int, float, List, np.ndarray)) or not isinstance(v_aprox, (int, float, List, np.ndarray)):
        return np.NaN

    if isinstance(v, List):
        v = np.array(v)
    if isinstance(v_aprox, List):
        v_aprox = np.array(v_aprox)

    try:
        return abs(v - v_aprox)
    except ValueError:
        return np.NaN


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[
    int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(v, (int, float, List, np.ndarray)) or not isinstance(v_aprox, (int, float, List, np.ndarray)):
        return np.NaN

    if isinstance(v, List):
        v = np.array(v)
        if np.any(v == 0):
            return np.NaN
    elif isinstance(v, np.ndarray):
        if np.any(v == 0):
            return np.NaN
    elif isinstance(v, (int, float)):
        if v == 0:
            return np.NaN

    if isinstance(v_aprox, List):
        v_aprox = np.array(v_aprox)

    try:
        return abs((v - v_aprox) / v)
    except ValueError:
        return np.NaN


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """

    if not isinstance(n, int) or not isinstance(c, (int, float)):
        return np.NaN

    P1 = 2 ** n - 2 ** n + c
    P2 = 2 ** n + c - 2 ** n

    return abs(P1 - P2)


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """

    if not isinstance(x, (int, float)) or not isinstance(n, int) or n <= 0:
        return np.NaN

    ex = 0

    for i in range(n):
        ex += x ** i / np.math.factorial(i)
    return ex


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """

    if not isinstance(k, int) or not isinstance(x, (int, float)) or k < 0:
        return np.NaN

    if k == 0:
        return 1
    elif k == 1:
        return np.cos(x)
    elif k > 0:
        return 2 * np.cos(x) * coskx1(k-1, x) - coskx1(k-2, x)
    #elif k < 0:
    #    return coskx1(-k, x)


def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(k, int) or not isinstance(x, (int, float)) or k < 0:
        return np.NaN

    if k == 1:
        return np.cos(x), np.sin(x)
    if k == 0:
        return 1, 0
    res = coskx2(k - 1, x)
    cosx = np.cos(x) * res[0] - np.sin(x) * res[1]
    sinx = np.sin(x) * res[0] + np.cos(x) * res[1]
    return cosx, sinx


def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """

    if not isinstance(n, int) or n <= 0:
        return np.NaN

    result = 0
    for i in range(1, n + 1):
        result+= 1 / i ** 2

    return np.sqrt(6 * result)
