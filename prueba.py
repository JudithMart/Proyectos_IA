import random
import math
from colorama import Fore, Style, init

init(autoreset=True)



def greedy_no_acumulable_con_backtracking(
    tablero,
    inicio,
    meta,
    permitir_diagonales=True,
    evitar_cortes_esquina=False,
    mostrar_minitablero=True, 
    parar_si_meta_vecina=True,
):
    filas, cols = len(tablero), len(tablero[0])

    def es_bloqueada(p):
        return tablero[p[1]][p[0]] == "X"

    # Usa tus vecinos pero filtrando según configuración
    def vecinos_local(p):
        x, y = p
        for v in vecinos(filas, cols, p):
            if not permitir_diagonales:
                # descarta diagonales
                if v[0] != x and v[1] != y:
                    continue
            # Evitar "corner cutting": si diagonal y ambos ortogonales bloqueados, descarta
            if evitar_cortes_esquina and (v[0] != x and v[1] != y):
                if tablero[y][v[0]] == "X" and tablero[v[1]][x] == "X":
                    continue
            yield v

    def ordenar_vecinos(p):
        cand = []
        for v in vecinos_local(p):
            if es_bloqueada(v):
                continue
            g = tablero[v[1]][v[0]]         # g no acumulativo
            h = heuristica(v, meta)
            cand.append((g + h, h, g, v))
        # Orden: f, luego h (más cerca a meta), luego g (más barato entrar)
        cand.sort(key=lambda t: (t[0], t[1], t[2]))
        return [v for _, _, _, v in cand]

    # Pila de (nodo_actual, vecinos_pendientes_ordenados)
    pila = [(inicio, ordenar_vecinos(inicio))]
    en_camino = {inicio}

    def pintar_estado(titulo):
        if not mostrar_minitablero:
            return
        print(titulo)
        camino_actual = [n for n, _ in pila]
        imprimir_tablero(tablero, inicio, meta, camino_actual)

    print(f"\nEstoy en  {inicio}:")
    pintar_estado("→ Tablero (inicio):")

    while pila:
        actual, cand = pila[-1]

        if actual == meta:
            # reconstruye desde la pila
            return [n for n, _ in pila]

        # descarta vecinos ya en el camino actual (evita ciclos)
        while cand and cand[0] in en_camino:
            cand.pop(0)

        if parar_si_meta_vecina and meta in cand:
            # Log como el de tus vecinos
            peso = tablero[meta[1]][meta[0]]
            h = 0.0
            print(f"  Vecino {meta} con peso={peso}")
            print(f"    g({meta}) = {peso}")
            print(f"    h({meta}) = sqrt(({meta[0]}-{meta[0]})²+({meta[1]}-{meta[1]})²) = {h:.2f}")
            print(f"    f({meta}) = g+h = {peso}+{h:.2f} = {peso+h:.2f}")
            en_camino.add(meta)
            pila.append((meta, []))
            pintar_estado("→ Tablero tras avanzar (meta vecina):")
            print(f"\nEstoy en  {meta}:")
            # Siguiente iteración devolverá el camino
            continue


        if not cand:
            # CALLEJÓN SIN SALIDA -> backtrack
            salgo, _ = pila.pop()
            en_camino.remove(salgo)
            if pila:
                print(f"Regresar: {salgo} -> {pila[-1][0]}")
                pintar_estado("↩︎ Tablero tras regresar:")
                print(f"\nEstoy en  {pila[-1][0]}:")
            continue

        siguiente = cand.pop(0)

        # Log de evaluación local
        peso = tablero[siguiente[1]][siguiente[0]]
        h = heuristica(siguiente, meta)
        print(f"  Vecino {siguiente} con peso={peso}")
        print(f"    g({siguiente}) = {peso}")
        print(f"    h({siguiente}) = sqrt(({siguiente[0]}-{meta[0]})²+({siguiente[1]}-{meta[1]})²) = {h:.2f}")
        print(f"    f({siguiente}) = g+h = {peso}+{h:.2f} = {peso+h:.2f}")

        # Avanzar
        en_camino.add(siguiente)
        pila.append((siguiente, ordenar_vecinos(siguiente)))
        pintar_estado("→ Tablero tras avanzar:")
        print(f"\nEstoy en  {siguiente}:")
    return None


# ===============================
# Funciones auxiliares
# ===============================

def generar_tablero(filas, columnas, num_celdas_cerradas):
    tablero = [[random.randint(1, filas*columnas) for _ in range(columnas)] for _ in range(filas)]

    cerradas = set()
    while len(cerradas) < num_celdas_cerradas:
        x, y = random.randint(0, columnas-1), random.randint(0, filas-1)  
        cerradas.add((x, y))

    for (x, y) in cerradas:
        tablero[y][x] = "X"  

    return tablero, cerradas

def imprimir_tablero(tablero, inicio, meta, camino=None):
    filas, columnas = len(tablero), len(tablero[0])
 
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
  
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def vecinos(filas, columnas, pos):
    direcciones = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]
    for dx, dy in direcciones:
        x2, y2 = pos[0]+dx, pos[1]+dy
        if 0 <= x2 < columnas and 0 <= y2 < filas:
            yield (x2, y2)

# ===============================
# MAIN
# ===============================

filas = int(input("Número de filas (eje Y): "))
columnas = int(input("Número de columnas (eje X): "))
num_cerradas = int(input("Número de celdas cerradas: "))

tablero, cerradas = generar_tablero(filas, columnas, num_cerradas)

print("\n=== Tablero inicial ===")
imprimir_tablero(tablero, (-1,-1), (-1,-1))  

sx, sy = map(int, input("Coordenadas de salida (x y): ").split())  
mx, my = map(int, input("Coordenadas de meta (x y): ").split())

while tablero[my][mx] == "X":
    print("La meta no puede ser una celda cerrada. Elige otra.")
    mx, my = map(int, input("Coordenadas de meta (x y): ").split())

inicio = (sx, sy)
meta = (mx, my)

camino = greedy_no_acumulable_con_backtracking(tablero, inicio, meta)


print("\n=== Resultado ===")
if camino:
    imprimir_tablero(tablero, inicio, meta, camino)
    print("Camino encontrado:", camino)
else:
    print("No existe camino posible")

