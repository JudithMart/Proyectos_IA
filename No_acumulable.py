import random
import math
from colorama import Fore, Style, init

init(autoreset=True)

# ===============================
# Funciones auxiliares
# ===============================

def generar_tablero(filas, columnas, num_celdas_cerradas):
    tablero = [[random.randint(1, filas*columnas) for _ in range(columnas)] for _ in range(filas)]

    cerradas = set()
    while len(cerradas) < num_celdas_cerradas:
        x, y = random.randint(0, columnas-1), random.randint(0, filas-1)  # x=columna, y=fila
        cerradas.add((x, y))

    for (x, y) in cerradas:
        tablero[y][x] = "X"   # tablero[fila][col]

    return tablero, cerradas

def imprimir_tablero(tablero, inicio, meta, camino=None):
    filas, columnas = len(tablero), len(tablero[0])
    # imprimimos de arriba hacia abajo, y=0 es arriba
    for y in range(filas):
        fila = []
        for x in range(columnas):
            if (x, y) == inicio:
                fila.append(Fore.BLUE + "S" + Style.RESET_ALL)
            elif (x, y) == meta:
                fila.append(Fore.YELLOW + "M" + Style.RESET_ALL)
            elif camino and (x, y) in camino:
                fila.append(Fore.GREEN + "*" + Style.RESET_ALL)
            elif tablero[y][x] == "X":
                fila.append(Fore.RED + "X" + Style.RESET_ALL)
            else:
                fila.append(str(tablero[y][x]))
        print("\t".join(fila))
    print()

def heuristica(a, b):
    # Distancia euclidiana
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def vecinos(filas, columnas, pos):
    direcciones = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]
    for dx, dy in direcciones:
        x2, y2 = pos[0]+dx, pos[1]+dy
        if 0 <= x2 < columnas and 0 <= y2 < filas:
            yield (x2, y2)

# ===============================
# Algoritmo A*
# ===============================

def a_star(tablero, inicio, meta):
    filas, columnas = len(tablero), len(tablero[0])
    abiertos = {inicio}
    cerrados = set()
    padres = {}

    g = {inicio: 0}
    f = {inicio: heuristica(inicio, meta)}

    while abiertos:
        actual = min(abiertos, key=lambda x: f.get(x, float("inf")))

        if actual == meta:
            camino = []
            while actual in padres:
                camino.append(actual)
                actual = padres[actual]
            camino.append(inicio)
            camino.reverse()
            return camino

        abiertos.remove(actual)
        cerrados.add(actual)

        print(f"\nExpandiendo nodo {actual}:")
        for vecino in vecinos(filas, columnas, actual):
            vx, vy = vecino
            if tablero[vy][vx] == "X" or vecino in cerrados:
                continue

            peso = tablero[vy][vx]
            h = heuristica(vecino, meta)
            f_valor = peso + h  # f = peso_actual + heurística

            if vecino not in abiertos or f_valor < f.get(vecino, float("inf")):
                padres[vecino] = actual
                g[vecino] = peso
                f[vecino] = f_valor
                abiertos.add(vecino)

                print(f"  Vecino {vecino} con peso={peso}")
                print(f"    g({vecino}) = {g[vecino]}")
                print(f"    h({vecino}) = sqrt(({vecino[0]}-{meta[0]})²+({vecino[1]}-{meta[1]})²) = {h:.2f}")
                print(f"    f({vecino}) = g+h = {f[vecino]:.2f}")

    return None

# ===============================
# MAIN
# ===============================

filas = int(input("Número de filas (eje Y): "))
columnas = int(input("Número de columnas (eje X): "))
num_cerradas = int(input("Número de celdas cerradas: "))

tablero, cerradas = generar_tablero(filas, columnas, num_cerradas)

print("\n=== Tablero inicial ===")
imprimir_tablero(tablero, (-1,-1), (-1,-1))  # aún sin inicio/meta

sx, sy = map(int, input("Coordenadas de salida (x y): ").split())  # (columna fila)
mx, my = map(int, input("Coordenadas de meta (x y): ").split())

while tablero[my][mx] == "X":
    print("La meta no puede ser una celda cerrada. Elige otra.")
    mx, my = map(int, input("Coordenadas de meta (x y): ").split())

inicio = (sx, sy)
meta = (mx, my)

camino = a_star(tablero, inicio, meta)

print("\n=== Resultado ===")
if camino:
    imprimir_tablero(tablero, inicio, meta, camino)
    print("Camino encontrado:", camino)
else:
    print("No existe camino posible")



# la casilla meta, no puede ser una casilla cerrada.
# Marcar de diferente manera la meta con una bandera o algo así 
# 
# Marcar al final el camino por otro color o algo así
#Suma total del camino
#
# Coordenas de salida a la meta   S(0,3) M(3,0)
#calcula la mejor ruta CONSIDERANDO LOS PESOS DE CADA CASILLA SIN CONSIDERAR LAS CASILLAS CERRADAS

# Tablero generado (X = obstáculo, S = salida, M = meta):
# 25 13  9  X  3 
# 12  M 10  X  7 
#  4  7  X  20  X 
# 10 22  8  X  15 
#  X 11  X  X  S 