import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional


class Almacen:
    def __init__(self):
        self.filas = 11
        self.columnas = 13
        self.layout = np.full((self.filas, self.columnas), ' ', dtype=object)

        # Diccionario de estanterías: producto -> (fila, col)
        self.estanterias = {
            1: (9, 2), 2: (9, 3),
            3: (8, 2), 4: (8, 3),
            5: (7, 2), 6: (7, 3),
            7: (6, 2), 8: (6, 3),

            9: (9, 6), 10: (9, 7),
            11: (8, 6), 12: (8, 7),
            13: (7, 6), 14: (7, 7),
            15: (6, 6), 16: (6, 7),

            17: (9, 10), 18: (9, 11),
            19: (8, 10), 20: (8, 11),
            21: (7, 10), 22: (7, 11),
            23: (6, 10), 24: (6, 11),

            25: (4, 2), 26: (4, 3),
            27: (3, 2), 28: (3, 3),
            29: (2, 2), 30: (2, 3),
            31: (1, 2), 32: (1, 3),

            33: (4, 6), 34: (4, 7),
            35: (3, 6), 36: (3, 7),
            37: (2, 6), 38: (2, 7),
            39: (1, 6), 40: (1, 7),

            41: (4, 10), 42: (4, 11),
            43: (3, 10), 44: (3, 11),
            45: (2, 10), 46: (2, 11),
            47: (1, 10), 48: (1, 11)
        }

        # Marca las estanterías en el layout
        for producto, pos in self.estanterias.items():
            fila, col = pos
            self.layout[fila, col] = str(producto)

        # Ubicación de la zona de carga
        self.carga_pos = (5, 0)
        self.layout[self.carga_pos] = 'C'

    def es_valido(self, pos: Tuple[int, int]) -> bool:
        """
        Verifica si una posición es válida para moverse (si es pasillo o no).

        """
        fila, col = pos
        # Verifica límites del almacén
        if fila < 0 or fila >= self.filas or col < 0 or col >= self.columnas:
            return False

        # Verifica si la posición contiene una estantería (un número) o está fuera de pasillos
        if self.layout[fila, col] not in [' ', 'C']:
            return False

        return True

    def get_posiciones_adyacentes(self, producto: int) -> List[Tuple[int, int]]:
        """
        Obtiene las posiciones *horizontalmente* adyacentes a una estantería.

        """
        if producto not in self.estanterias:
            return []

        fila, col = self.estanterias[producto]

        # Solo posiciones a izquierda y derecha (horizontalmente adyacentes)
        adyacentes = [
            (fila, col - 1),
            (fila, col + 1)
        ]

        # Filtra solo las posiciones válidas (pasillos)
        return [pos for pos in adyacentes if self.es_valido(pos)]


class AgenteAEstrella:
    def __init__(self, almacen: Almacen):
        """
        Inicializa el agente con una referencia al Almacen.
        """
        self.almacen = almacen

    def heuristica(self, pos: Tuple[int, int], destino: Tuple[int, int]) -> float:
        """
        Calcula la heurística (distancia Manhattan) entre dos posiciones.
        """
        return abs(pos[0] - destino[0]) + abs(pos[1] - destino[1])

    def encontrar_camino(self, inicio: Optional[Tuple[int, int]], destino_producto: int) -> List[Tuple[int, int]]:
        """
        Encuentra el camino más corto desde 'inicio' hasta alguna de las posiciones
        horizontalmente adyacentes al 'destino_producto' usando A*.
        """
        # Si no se proporciona posición inicial, usar la zona de carga
        if inicio is None:
            inicio = self.almacen.carga_pos

        # Obtiene posiciones adyacentes válidas al producto destino
        destinos_posibles = self.almacen.get_posiciones_adyacentes(destino_producto)

        if not destinos_posibles:
            print(f"No hay posiciones adyacentes accesibles para el producto {destino_producto}.")
            return []

        # Lista abierta para el algoritmo A* (cola de prioridad)
        # Cada elemento de la cola: (f, contador, posicion, padre, g)
        abierta = []
        contador = 0
        """
         Calcular la heurística inicial como la mínima distancia Manhattan
         desde la posición de inicio a cualquiera de las celdas de destino

        """
        g_inicio = 0
        h_inicio = min(self.heuristica(inicio, d) for d in destinos_posibles)
        f_inicio = g_inicio + h_inicio

        heapq.heappush(abierta, (f_inicio, contador, inicio, None, g_inicio))

        # Diccionario para rastrear los nodos visitados y su costo
        # cerrada[pos] = (g, padre)
        cerrada = {}

        while abierta:
            # Extrae el nodo con menor f de la lista abierta
            _, _, pos_actual, padre, g_actual = heapq.heappop(abierta)

            # Verifica si ya se procesó pos_actual con un mejor costo g
            if pos_actual in cerrada and cerrada[pos_actual][0] <= g_actual:
                continue

            # Marca la posición actual como visitada con su costo y padre
            cerrada[pos_actual] = (g_actual, padre)

            # Verifica si la posición actual es uno de los destinos
            if pos_actual in destinos_posibles:
                # Reconstruye el camino
                camino = []
                while pos_actual is not None:
                    camino.append(pos_actual)
                    pos_actual = cerrada[pos_actual][1]
                return camino[::-1]  # Invierte para obtener el camino de inicio a fin

            # Explora los vecinos (arriba, abajo, izquierda, derecha)
            movimientos = [
                (pos_actual[0] - 1, pos_actual[1]),  # Arriba
                (pos_actual[0] + 1, pos_actual[1]),  # Abajo
                (pos_actual[0], pos_actual[1] - 1),  # Izquierda
                (pos_actual[0], pos_actual[1] + 1)  # Derecha
            ]

            for siguiente_pos in movimientos:
                # Verifica si el movimiento es válido
                if not self.almacen.es_valido(siguiente_pos):
                    continue

                # Calcula el costo acumulado
                g_siguiente = g_actual + 1  # Costo uniforme

                # Verifica si ya se visitó con un costo menor
                if siguiente_pos in cerrada and cerrada[siguiente_pos][0] <= g_siguiente:
                    continue

                # Calcula heurística hacia el destino más cercano
                h_siguiente = min(self.heuristica(siguiente_pos, d) for d in destinos_posibles)
                f_siguiente = g_siguiente + h_siguiente

                contador += 1
                heapq.heappush(abierta, (f_siguiente, contador, siguiente_pos, pos_actual, g_siguiente))

        # Si la lista abierta se vacía sin encontrar destino, no hay camino
        print("No se encontró un camino válido.")
        return []

    def visualizar_camino(self, camino: List[Tuple[int, int]], destino_producto: int) -> None:
        if not camino:
            print("No hay camino para visualizar.")
            return

        visualizacion = np.zeros((self.almacen.filas, self.almacen.columnas))

        # Códigos para visualización:
        # 0: Pasillo vacío
        # 1: Estantería
        # 2: Camino
        # 3: Inicio
        # 4: Fin
        # 5: Zona de carga

        # Marca estanterías
        for producto, pos in self.almacen.estanterias.items():
            visualizacion[pos] = 1

        # Marca zona de carga
        visualizacion[self.almacen.carga_pos] = 5

        # Marca camino
        for i, pos in enumerate(camino):
            if i == 0:
                visualizacion[pos] = 3
            elif i == len(camino) - 1:
                visualizacion[pos] = 4
            else:
                visualizacion[pos] = 2

        # Gráfica
        plt.figure(figsize=(12, 10))
        cmap = plt.cm.colors.ListedColormap(['white', 'lightgray', 'lightblue', 'green', 'red', 'yellow'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(visualizacion, cmap=cmap, norm=norm)

        # Dibuja la cuadrícula
        plt.grid(color='black', linestyle='-', linewidth=0.5)

        # Agrega coordenadas numéricas en cada celda (fila,columna)
        for f in range(self.almacen.filas):
            for c in range(self.almacen.columnas):
                # Si es una estantería o la zona de carga, no mostrar coordenadas
                if visualizacion[f, c] != 1 and visualizacion[f, c] != 5:
                    plt.text(c, f, f"({f},{c})", ha='center', va='center',
                             color='black', fontsize=7, alpha=0.5)

        # Agrega etiquetas de productos
        for producto, (f, c) in self.almacen.estanterias.items():
            plt.text(c, f, str(producto), ha='center', va='center', color='black', fontsize=9, fontweight='bold')

        # Etiqueta zona de carga
        plt.text(self.almacen.carga_pos[1], self.almacen.carga_pos[0], 'C',
                 ha='center', va='center', color='black', fontweight='bold')

        # Configuración de ejes
        plt.xticks(np.arange(-0.5, self.almacen.columnas, 1), [])
        plt.yticks(np.arange(-0.5, self.almacen.filas, 1), [])
        plt.xlim(-0.5, self.almacen.columnas - 0.5)
        plt.ylim(-0.5, self.almacen.filas - 0.5)

        # Título y leyenda
        if camino[0] == self.almacen.carga_pos:
            origen = "zona de carga (C)"
        else:
            origen = f"posición (fila={camino[0][0]}, col={camino[0][1]})"

        plt.title(f"Camino desde {origen} hasta el producto {destino_producto}\n"
                  f"Longitud del camino: {len(camino) - 1} pasos", fontsize=12)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Pasillo'),
            Patch(facecolor='lightgray', edgecolor='black', label='Estantería'),
            Patch(facecolor='lightblue', edgecolor='black', label='Camino'),
            Patch(facecolor='green', edgecolor='black', label='Inicio'),
            Patch(facecolor='red', edgecolor='black', label='Fin'),
            Patch(facecolor='yellow', edgecolor='black', label='Zona de carga')
        ]
        plt.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.05), ncol=3)

        plt.tight_layout()
        plt.show()

        # Imprime pasos en consola
        print(f"\nCamino desde {origen} hasta el producto {destino_producto}:")
        print(f"Longitud total: {len(camino) - 1} pasos")
        print("\nPasos:")
        for i, pos in enumerate(camino):
            if i == 0:
                print(f"Posición inicial: Fila {pos[0]}, Columna {pos[1]}")
            else:
                print(f"Paso {i}: Fila {pos[0]}, Columna {pos[1]}")


def main():
    almacen = Almacen()
    agente = AgenteAEstrella(almacen)

    while True:
        # Pide número de estantería
        entrada_producto = input("Ingrese el número de estantería a buscar (1-48) o 0 para salir: ")
        try:
            producto_destino = int(entrada_producto)
        except ValueError:
            print("Por favor, ingrese un número válido.")
            continue

        if producto_destino == 0:
            print("Saliendo del programa...")
            break

        if not (1 <= producto_destino <= 48):
            print("El número de estantería debe estar entre 1 y 48.")
            continue

        # Pide posición inicial
        inicio_input = input("Ingrese la posición inicial (fila,columna) o presione Enter para la zona de carga: ")
        if inicio_input.strip():
            try:
                fila_str, col_str = inicio_input.split(',')
                fila_in = int(fila_str)
                col_in = int(col_str)
                inicio = (fila_in, col_in)
            except Exception:
                print("Formato inválido. Use fila,columna (ej. 5,3). Se usará la zona de carga por defecto.")
                inicio = None
        else:
            inicio = None

        # Busca camino
        camino = agente.encontrar_camino(inicio, producto_destino)
        # Visualiza resultado
        agente.visualizar_camino(camino, producto_destino)


if __name__ == "__main__":
    main()