from typing import List

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import string
import random


def compare_plot(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
                 xlabel: str = None, ylabel: str = None, title: str = None, label1: str = "function 1",
                 label2: str = "function 2"):
    """Funkcja służąca do porównywania dwóch wykresów typu plot.
    Szczegółowy opis w zadaniu 3.

    Parameters:
    x1(np.ndarray): wektor wartości osi x dla pierwszego wykresu,
    y1(np.ndarray): wektor wartości osi y dla pierwszego wykresu,
    x2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    y2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    xlabel(str): opis osi x,
    ylabel(str): opis osi y,
    title(str): tytuł wykresu ,
    label1(str): nazwa serii z pierwszego wykresu,
    label2(str): nazwa serii z drugiego wykresu.

    Returns:
    matplotlib.pyplot.figure: wykres zbiorów (x1,y1), (x2,y2) zgody z opisem z zadania 3
    """

    if x1.shape != y1.shape or x2.shape != y2.shape or x1.shape[0] < 2 or x1.shape[0] < 2:
        return None
    # uznałam, że jeżeli da się wyrysować tylko jeden wykres, to lepiej, żeby zwracało none, bo to w końcu ma
    # porównywać, a nie da się porównać czegoś z niczym

    fig, ax = plt.subplots()
    ax.plot(x1, y1, "-b", linewidth=4, label=label1)
    ax.plot(x2, y2, "-r", linewidth=2, label=label2)

    plt.legend([label1] + [label2], loc='best')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()
    return fig


def parallel_plot(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
                  x1label: str = None, y1label: str = None, x2label: str = None, y2label: str = None, title: str = None, orientation: str ="-"):
    """Funkcja służąca do stworzenia dwóch wykresów typu plot w konwencji subplot wertykalnie lub chorycontalnie.
    Szczegółowy opis w zadaniu 5.

    Parameters:
    x1(np.ndarray): wektor wartości osi x dla pierwszego wykresu,
    y1(np.ndarray): wektor wartości osi y dla pierwszego wykresu,
    x2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    y2(np.ndarray): wektor wartości osi x dla drugiego wykresu,
    x1label(str): opis osi x dla pierwszego wykresu,
    y1label(str): opis osi y dla pierwszego wykresu,
    x2label(str): opis osi x dla drugiego wykresu,
    y2label(str): opis osi y dla drugiego wykresu,
    title(str): tytuł wykresu,
    orientation(str): parametr przyjmujący wartość '-' jeżeli subplot ma posiadać dwa wiersze albo '|' jeżeli ma posiadać dwie kolumny.

    Returns:
    matplotlib.pyplot.figure: wykres zbiorów (x1,y1), (x2,y2) zgody z opisem z zadania 5
    """
    if x1.shape != y1.shape or x2.shape != y2.shape or x1.shape[0] < 2 or x1.shape[0] < 2:
        return None
    # podobnie jak w powyższej funkcji, paraller_plot będzie zwracał None, nawet jeśli jedną funkcję da się wyrysować

    #if len(np.unique(x1)) != len(x1) or len(np.unique(x2)) != len(x2):
        #return None
    # dwie powyższe linijki są potrzebne jedynie do przejścia testu, który nie rozumiem dlaczego powinien zwracac none, w końcy np taka
    #spirala ma parę wartości dla jednego x

    if orientation == "-":
        fig, (ax1, ax2) = plt.subplots(2, 1)
    elif orientation == "|":
        fig, (ax1, ax2) = plt.subplots(1, 2)
    else:
        return None

    fig.suptitle(title)
    ax1.plot(x1, y1)
    ax2.plot(x2, y2)
    fig.tight_layout(pad=3.0)
    ax1.set(xlabel=x1label, ylabel=y1label)
    ax2.set(xlabel=x2label, ylabel=y2label)

    plt.show()
    return fig


def log_plot(x: np.ndarray, y: np.ndarray, xlabel: str = None, ylabel: str = None, title: str = None,
             log_axis: str = 'y'):
    """Funkcja służąca do tworzenia wykresów ze skalami logarytmicznymi.
    Szczegółowy opis w zadaniu 7.

    Parameters:
    x(np.ndarray): wektor wartości osi x,
    y(np.ndarray): wektor wartości osi y,
    xlabel(str): opis osi x,
    ylabel(str): opis osi y,
    title(str): tytuł wykresu ,
    log_axis(str): wartość oznacza:
        - 'x' oznacza skale logarytmiczną na osi x,
        - 'y' oznacza skale logarytmiczną na osi y,
        - 'xy' oznacza skale logarytmiczną na obu osiach.

    Returns:
    matplotlib.pyplot.figure: wykres zbiorów (x,y) zgody z opisem z zadania 7
    """

    if x.shape != y.shape or x.shape[0] < 2:
        return None

    fig, ax = plt.subplots()
    plt.plot(x, y)

    if log_axis == "x":
        ax.set_xscale('log')
    elif log_axis == "y":
        ax.set_yscale('log')
    elif log_axis == "xy":
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        return None

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.grid()
    plt.show()
    return fig
