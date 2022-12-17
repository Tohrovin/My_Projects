import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle
from inspect import isfunction

from typing import Union, List, Tuple


def fun(x):
    return np.exp(-2 * x) + x ** 2 - 1


def dfun(x):
    return -2 * np.exp(-2 * x) + 2 * x


def ddfun(x):
    return 4 * np.exp(-2 * x) + 2


def bisection(a: Union[int, float], b: Union[int, float], f: typing.Callable[[float], float], epsilon: float,
              iteration: int) -> Tuple[float, int]:
    """funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    """

    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isfunction(f):
        return None
    if not isinstance(epsilon, float) or not isinstance(iteration, int) or iteration <= 0 or epsilon < 0:
        return None
    if f(a) * f(b) >= 0:
        return None

    for i in range(0, iteration):
        c = (a + b) / 2
        fc = f(c)
        if f(a) * fc <= 0:
            b = c
        elif f(b) * fc <= 0:
            a = c
        else:
            return None

        if np.abs(c) <= epsilon or np.abs(f(c)) <= epsilon:
            return c, i

    return a, iteration


def secant(a: Union[int, float], b: Union[int, float], f: typing.Callable[[float], float], epsilon: float,
           iteration: int) -> Tuple[float, int]:
    """funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    """

    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isfunction(f):
        return None
    if not isinstance(epsilon, float) or not isinstance(iteration, int) or iteration <= 0 or epsilon < 0:
        return None
    if f(a) * f(b) >= 0:
        return None

    fa = f(a)
    fb = f(b)

    for i in range(iteration):

        fa = f(a)
        fb = f(b)

        c = (fb * a - fa * b) / (fb - fa)
        fc = f(c)

        if fa * fc <= 0:
            b = c
        elif fa * fc > 0:
            a = c

        if abs(b - a) < epsilon or abs(fc) < epsilon:
            return c, i

    return (fb * a - fa * b) / (fb - fa), iteration


def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float],
           ddf: typing.Callable[[float], float], a: Union[int, float], b: Union[int, float], epsilon: float,
           iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if not isfunction(f) or not isfunction(df) or not isfunction(ddf) or iteration <= 0 or epsilon < 0:
        return None

    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(epsilon,
                                                                                            float) or not isinstance(
            iteration, int):
        return None

    if f(a) * f(b) >= 0:
        return None

    a_b = np.linspace(a, b, 100)
    df_val = df(a_b)
    ddf_val = ddf(a_b)
    if not ((np.all(np.sign(df_val) < 0) or np.all(np.sign(df_val) > 0)) and
            (np.all(np.sign(ddf_val) < 0) or np.all(np.sign(ddf_val) > 0))):
        return None



    c = (a + b) / 2

    for i in range(iteration):
        fc = f(c)
        dfc = df(c)
        if dfc == 0:
            return None
        else:
            c = c - fc / dfc

        if np.abs(fc) < epsilon:
            return c, i

    return c, iteration
