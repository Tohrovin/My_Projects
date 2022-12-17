import numpy as np
from numpy.core.fromnumeric import transpose
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if not isinstance(m, int) or m <= 0:
        return None

    A, b = np.random.randint(9, size=(m, m)), np.random.randint(9, size=m)

    for i in range(m):
        sum = 0

        for j in range(m):

            if i != j:
                sum += A[i][j] + A[j][i]

        A[i][i] += sum + 1

    return A, b


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(A, np.ndarray):
        return None

    try:
        if np.shape(A)[0] != np.shape(A)[1]:
            return None
    except IndexError:
        return None

    size = np.shape(A)[0]
    for i in range(size):
        sum1 = 0
        sum2 = 0
        for j in range(size):
            if i != j:
                sum1 += A[i][j]
                sum2 += A[j][i]

        if A[i][i] < sum1 or A[i][i] < sum2:
            return False

    return True


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(m, int) or m <= 0:
        return None

    A, b = np.random.randint(9, size=(m, m)), np.random.randint(9, size=m)

    A = (A + np.transpose(A)) / 2

    return (A, b)


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if not isinstance(A, np.ndarray):
        return None

    try:
        if np.shape(A)[0] != np.shape(A)[1]:
            return None
    except IndexError:
        return None

    m = np.shape(A)[0]

    for i in range(m):
        for j in range(m):
            if A[i][j] != A[j][i]:
                return False

    return True


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray) or not isinstance(x_init, np.ndarray):
        return None
    if not isinstance(epsilon, float) and epsilon is not None:
        return None
    if not isinstance(maxiter, int) and maxiter is not None:
        return None

    try:
        if np.shape(A)[0] != np.shape(A)[1] or np.shape(A)[0] != np.shape(b)[0] or np.shape(A)[0] != np.shape(x_init)[0]:
            return None
    except IndexError:
        return None

    D = np.diag(np.diag(A))
    LU = A - D
    x = x_init
    D_inv = np.diag(1 / np.diag(D))
    iter = 0

    for i in range(maxiter):
        x_new = np.dot(D_inv, b - np.dot(LU, x))
        r_norm = np.linalg.norm(x_new - x)
        iter += 1

        if r_norm < epsilon:
            return x_new, iter
        x = x_new
    return x, iter

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    if np.shape(x)[0] != np.shape(A)[1] or np.shape(b)[0] != np.shape(A)[0]:
        return None

    return np.linalg.norm(b - A @ np.transpose(x))
