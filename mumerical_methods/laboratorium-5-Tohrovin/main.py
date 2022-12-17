import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple


def first_spline(x: np.ndarray, y: np.ndarray):
    """Funkcja wyznaczająca wartości współczynników spline pierwszego stopnia.

    Parametrs:
    x(float): argumenty, dla danych punktów
    y(float): wartości funkcji dla danych argumentów

    return (a,b) - krotka zawierająca współczynniki funkcji linowych"""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or np.size(x) != np.size(y):
        return None

    an = []
    bn = []

    for i in range(np.size(x) - 1):
        an.append((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
        bn.append(y[i] - an[i] * x[i])

    return an, bn


def jacobi(A, b, x0, tol, n_iterations=200):
    """
    Iteracyjne rozwiązanie równania Ax=b dla zadanego x0

    Returns:
    x - estymowane rozwiązanie
    """

    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    counter = 0
    x_diff = tol + 1

    while (x_diff > tol) and (counter < n_iterations):  # iteration level
        for i in range(0, n):  # element wise level for x
            s = 0
            for j in range(0, n):  # summation for i !=j
                if i != j:
                    s += A[i, j] * x_prev[j]

            x[i] = (b[i] - s) / A[i, i]
        # update values
        counter += 1
        x_diff = (np.sum((x - x_prev) ** 2)) ** 0.5
        x_prev = x.copy()  # use new x for next iteration

    print("Number of Iterations: ", counter)
    print("Norm of Difference: ", x_diff)
    return x


def cubic_spline(x: np.ndarray, y: np.ndarray, tol=1e-100):
    """
    Interpolacja splajnów cubicznych

    Returns:
    b współczynnik przy x stopnia 1
    c współczynnik przy x stopnia 2
    d współczynnik przy x stopnia 3
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or np.size(x) != np.size(y):
        return None

    ### check if sorted
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    size = len(x)
    delta_x = np.diff(x)
    delta_y = np.diff(y)

    ### Get matrix A
    A = np.zeros(shape=(size, size))
    b = np.zeros(shape=(size, 1))
    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, size - 1):
        A[i, i - 1] = delta_x[i - 1]
        A[i, i + 1] = delta_x[i]
        A[i, i] = 2 * (delta_x[i - 1] + delta_x[i])
        ### Get matrix b
        b[i, 0] = 3 * (delta_y[i] / delta_x[i] - delta_y[i - 1] / delta_x[i - 1])

    ### Solves for c in Ac = b
    print('Jacobi Method Output:')
    c = jacobi(A, b, np.zeros(len(A)), tol=tol, n_iterations=1000)

    ### Solves for d and b
    d = np.zeros(shape=(size - 1, 1))
    b = np.zeros(shape=(size - 1, 1))
    for i in range(0, len(d)):
        d[i] = (c[i + 1] - c[i]) / (3 * delta_x[i])
        b[i] = (delta_y[i] / delta_x[i]) - (delta_x[i] / 3) * (2 * c[i] + c[i + 1])

    return b.squeeze(), c.squeeze(), d.squeeze()

    # funkcje z poprzednich labów:


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

    elif isinstance(xr, List) and isinstance(x, List):
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

    return np.NaN