import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.linalg
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P
import pickle


# zad1
def polly_A(x: np.ndarray):
    """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    Parameters:
    x: wektor pierwiastków
    Results:
    (np.ndarray): wektor współczynników
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if not isinstance(x, np.ndarray):
        return None

    return P.polyfromroots(x)


def roots_20(a: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if not isinstance(a, np.ndarray):
        return None

    length = np.size(a)

    x = np.linspace(1, length, length)

    err = np.random.random_sample(length) / 1e10
    a = a + err

    roots = np.sort(P.polyroots(a))

    return a, roots


# zad 2

def frob_a(wsp: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots

                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if not isinstance(wsp, np.ndarray):
        return None

    left = np.zeros((wsp.size - 1, 1))
    frob_m = np.eye(wsp.size - 1)

    frob_m = np.c_[left, frob_m]

    frob_m = np.concatenate((frob_m, np.reshape(-wsp, (1, wsp.shape[0]))), axis=0)

    return frob_m, np.linalg.eigvals(frob_m), scipy.linalg.schur(frob_m), np.sort(P.polyroots(wsp))
