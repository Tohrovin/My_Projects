import numpy as np
import pickle

from typing import Union, List, Tuple
import numpy.linalg as nplin


def random_matrix_Ab(m: int, range: int = 500):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
            Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m, int) and m > 0:
        return np.random.randint(range, size=(m, m)), np.random.randint(range, size=m)
    else:
        return None


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b
    Parameters:
    A: macierz A (m,m) zawierająca współczynniki równania 
    x: wektor x (m.) zawierający rozwiązania równania 
    b: wektor b (m,) zawierający współczynniki po prawej stronie równania
    Results:
    (float)- wartość normy residuom dla podanych parametrów"""

    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    if np.shape(A)[0] != np.shape(A)[1] or np.shape(x)[0] != np.shape(A)[0] or np.shape(b) != np.shape(x):
        return None

    return nplin.norm(b - A @ np.transpose(x))


def log_sing_value(n: int, min_order: Union[int, float], max_order: Union[int, float]):
    """Funkcja generująca wektor wartości singularnych rozłożonych w skali logarytmiczne
    
        Parameters:
         n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
         min_order(int,float): rząd najmniejszej wartości w wektorze wartości singularnych
         max_order(int,float): rząd największej wartości w wektorze wartości singularnych
         Results:
         np.ndarray - wektor nierosnących wartości logarytmicznych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
         """

    if not isinstance(n, int) or not isinstance(min_order, (int, float)) or not isinstance(max_order, (
    int, float)) or min_order > max_order or n <= 0:
        return None
    else:
        return np.flip(np.logspace(min_order, max_order, n))


def order_sing_value(n: int, order: Union[int, float] = 2, site: str = 'gre'):
    """Funkcja generująca wektor losowych wartości singularnych (n,) będących wartościami zmiennoprzecinkowymi losowanymi przy użyciu funkcji np.random.rand(n)*10. 
        A następnie ustawiająca wartość minimalną (site = 'low') albo maksymalną (site = 'gre') na wartość o  10**order razy mniejszą/większą.
    
        Parameters:
        n(np.ndarray): rozmiar wektora wartości singularnych (n,), gdzie n>0
        order(int,float): rząd przeskalowania wartości skrajnej
        site(str): zmienna wskazująca stronnę zmiany:
            - site = 'low' -> sing_value[-1] * 10**order
            - site = 'gre' -> sing_value[0] * 10**order
        
        Results:
        np.ndarray - wektor wartości singularnych o wymiarze (n,) zawierający wartości logarytmiczne na zadanym przedziale
        """
    if not isinstance(n, int) or not isinstance(order, (int, float)) or not isinstance(site, str) or n <= 0:
        return None

    else:
        sing = np.random.rand(n) * 10

        if site == "low":
            max_val = np.where(sing == np.amax(sing))
            sing[max_val] = sing[max_val] * (10 ** order)
        elif site == "gre":
            min_val = np.where(sing == np.amin(sing))
            sing[min_val] = sing[min_val] / (10 ** order)
        else:
            return None

        return np.flip(np.sort(sing))


def create_matrix_from_A(A: np.ndarray, sing_value: np.ndarray):
    """Funkcja generująca rozkład SVD dla macierzy A i zwracająca otworzenie macierzy A z wykorzystaniem zdefiniowanego wektora warości singularnych
            Parameters:
            A(np.ndarray): rozmiarz macierzy A (m,m)
            sing_value(np.ndarray): wektor wartości singularnych (m,)
            Results:
            np.ndarray: macierz (m,m) utworzoną na podstawie rozkładu SVD zadanej macierzy A z podmienionym wektorem wartości singularnych na wektor sing_valu """

    if not isinstance(A, np.ndarray) or not isinstance(sing_value, np.ndarray):
        return None
    if np.shape(A)[0] != np.shape(A)[1] or np.shape(sing_value)[0] != np.shape(A)[0]:
        return None

    U, S, V = nplin.svd(A)

    return np.dot(U * sing_value, V)
