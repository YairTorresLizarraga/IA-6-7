#TORRES LIZARRAGA YAIR
#Puzzle 24 

import random
import math
import time
from typing import List, Tuple

TAMANO = 5

ESTADO_OBJETIVO = tuple(list(range(1, 25)) + [0])

POSICION_OBJETIVO = {
    ESTADO_OBJETIVO[i]: (i // TAMANO, i % TAMANO)
    for i in range(25)
}

nodos_expandidos = 0


def heuristica_manhattan(estado: Tuple[int]) -> int:
    distancia_total = 0

    for indice, ficha in enumerate(estado):
        if ficha != 0:
            fila_actual = indice // TAMANO
            columna_actual = indice % TAMANO
            fila_objetivo, columna_objetivo = POSICION_OBJETIVO[ficha]

            distancia_total += abs(fila_actual - fila_objetivo) + \
                               abs(columna_actual - columna_objetivo)

    return distancia_total


def heuristica_conflicto_lineal(estado: Tuple[int]) -> int:
    distancia = heuristica_manhattan(estado)
    penalizacion = 0

    for fila in range(TAMANO):
        fichas_fila = []

        for columna in range(TAMANO):
            ficha = estado[fila * TAMANO + columna]
            if ficha != 0 and POSICION_OBJETIVO[ficha][0] == fila:
                fichas_fila.append(ficha)

        for i in range(len(fichas_fila)):
            for j in range(i + 1, len(fichas_fila)):
                if POSICION_OBJETIVO[fichas_fila[i]][1] > \
                   POSICION_OBJETIVO[fichas_fila[j]][1]:
                    penalizacion += 2

    for columna in range(TAMANO):
        fichas_columna = []

        for fila in range(TAMANO):
            ficha = estado[fila * TAMANO + columna]
            if ficha != 0 and POSICION_OBJETIVO[ficha][1] == columna:
                fichas_columna.append(ficha)

        for i in range(len(fichas_columna)):
            for j in range(i + 1, len(fichas_columna)):
                if POSICION_OBJETIVO[fichas_columna[i]][0] > \
                   POSICION_OBJETIVO[fichas_columna[j]][0]:
                    penalizacion += 2

    return distancia + penalizacion


def es_resoluble(estado: Tuple[int]) -> bool:
    inversiones = 0
    arreglo = [x for x in estado if x != 0]

    for i in range(len(arreglo)):
        for j in range(i + 1, len(arreglo)):
            if arreglo[i] > arreglo[j]:
                inversiones += 1

    fila_vacio = estado.index(0) // TAMANO

    if TAMANO % 2 == 1:
        return inversiones % 2 == 0
    else:
        return (inversiones + fila_vacio) % 2 == 1


def obtener_vecinos(estado: Tuple[int]) -> List[Tuple[int]]:
    vecinos = []
    indice_cero = estado.index(0)
    fila = indice_cero // TAMANO
    columna = indice_cero % TAMANO

    movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for df, dc in movimientos:
        nueva_fila = fila + df
        nueva_columna = columna + dc

        if 0 <= nueva_fila < TAMANO and 0 <= nueva_columna < TAMANO:
            nuevo_indice = nueva_fila * TAMANO + nueva_columna
            nuevo_estado = list(estado)

            nuevo_estado[indice_cero], nuevo_estado[nuevo_indice] = \
                nuevo_estado[nuevo_indice], nuevo_estado[indice_cero]

            vecinos.append(tuple(nuevo_estado))

    return vecinos


def busqueda_ida(estado_inicial: Tuple[int], heuristica):
    global nodos_expandidos

    nodos_expandidos = 0
    inicio = time.time()

    umbral = heuristica(estado_inicial)
    camino = [estado_inicial]

    while True:
        resultado = busqueda_limitada(camino, 0, umbral, heuristica)

        if resultado == "ENCONTRADO":
            fin = time.time()
            return {
                "camino": camino.copy(),
                "nodos": nodos_expandidos,
                "tiempo": (fin - inicio) * 1000,
                "longitud": len(camino) - 1
            }

        if resultado == math.inf:
            return None

        umbral = resultado


def busqueda_limitada(camino, costo_actual, umbral, heuristica):
    global nodos_expandidos

    nodo = camino[-1]
    costo_estimado = costo_actual + heuristica(nodo)

    if costo_estimado > umbral:
        return costo_estimado

    if nodo == ESTADO_OBJETIVO:
        return "ENCONTRADO"

    minimo = math.inf
    nodos_expandidos += 1 

    for vecino in obtener_vecinos(nodo):
        if vecino not in camino:
            camino.append(vecino)

            resultado = busqueda_limitada(
                camino,
                costo_actual + 1,
                umbral,
                heuristica
            )

            if resultado == "ENCONTRADO":
                return "ENCONTRADO"

            if resultado < minimo:
                minimo = resultado

            camino.pop()

    return minimo


def generar_estado_aleatorio(movimientos=15):
    estado = ESTADO_OBJETIVO
    for _ in range(movimientos):
        estado = random.choice(obtener_vecinos(estado))
    return estado


def imprimir_tablero(estado):
    for i in range(TAMANO):
        fila = estado[i * TAMANO:(i + 1) * TAMANO]
        print(" ".join(f"{x:2}" if x != 0 else "  " for x in fila))
    print()


def leer_estado_manual():
    print("Introduce 25 numeros del 0 al 24 separados por espacio:")
    entrada = input("Estado: ").strip().split()

    if len(entrada) != 25:
        print("Error: debes introducir exactamente 25 numeros.")
        return None

    estado = tuple(int(x) for x in entrada)

    if set(estado) != set(range(25)):
        print("Error: los numeros deben ir del 0 al 24 sin repetir.")
        return None

    return estado


if __name__ == "__main__":

    print("1 - Generar estado aleatorio")
    print("2 - Introducir estado manual")
    opcion_estado = input("Elige opcion: ").strip()

    if opcion_estado == "2":
        estado_inicial = leer_estado_manual()
        if estado_inicial is None:
            exit()
    else:
        estado_inicial = generar_estado_aleatorio()

    print("\n1 - Usar heuristica Manhattan")
    print("2 - Usar heuristica Conflicto lineal")
    opcion_heuristica = input("Elige heuristica: ").strip()

    if opcion_heuristica == "2":
        heuristica = heuristica_conflicto_lineal
    else:
        heuristica = heuristica_manhattan

    print("\nEstado inicial:")
    imprimir_tablero(estado_inicial)

    if not es_resoluble(estado_inicial):
        print("El tablero no es resoluble.")
    else:
        print("Resolviendo...\n")
        resultado = busqueda_ida(estado_inicial, heuristica)

        if resultado:
            print("Resuelto en", resultado["longitud"], "movimientos\n")

            print("RENDIMIENTO")
            print("Nodos expandidos:", resultado["nodos"])
            print("Tiempo de ejecución:", round(resultado["tiempo"], 2), "milisegundos\n")

            for paso in resultado["camino"]:
                imprimir_tablero(paso)
        else:
            print("No se encontro solucion.")
