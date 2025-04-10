import random
import math
import matplotlib
import heapq
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import matplotlib.animation as animation
from matplotlib.patches import Patch
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

    def es_valido(self, pos: tuple) -> bool:
        """Verifica si una posición es válida para moverse."""
        fila, col = pos
        if fila < 0 or fila >= self.filas or col < 0 or col >= self.columnas:
            return False
        if self.layout[fila, col] not in [' ', 'C']:
            return False
        return True

    def get_posiciones_adyacentes(self, producto: int) -> list:
        """Obtiene las posiciones adyacentes a una estantería."""
        if producto not in self.estanterias:
            return []
        fila, col = self.estanterias[producto]
        adyacentes = [(fila, col - 1), (fila, col + 1)]
        return [pos for pos in adyacentes if self.es_valido(pos)]


class AgenteAEstrella:
    def __init__(self, almacen: Almacen):
        self.almacen = almacen

    def heuristica(self, pos: Tuple[int, int], destino: Tuple[int, int]) -> float:
        return abs(pos[0] - destino[0]) + abs(pos[1] - destino[1])

    def encontrar_camino(self, inicio: Optional[Tuple[int, int]], destino_producto: Optional[int]) -> List[
        Tuple[int, int]]:
        if inicio is None:
            inicio = self.almacen.carga_pos

        # Si destino_producto es None, vamos directamente a la zona de carga
        if destino_producto is None:
            destino = self.almacen.carga_pos
            destinos_posibles = [destino]  # Solo hay un destino posible: la zona de carga
        else:
            # Si es un producto normal, obtenemos las posiciones adyacentes
            destinos_posibles = self.almacen.get_posiciones_adyacentes(destino_producto)
            if not destinos_posibles:
                print(f"No hay posiciones adyacentes accesibles para el producto {destino_producto}.")
                return []

        abierta = []
        contador = 0
        g_inicio = 0
        h_inicio = min(self.heuristica(inicio, d) for d in destinos_posibles)
        f_inicio = g_inicio + h_inicio

        heapq.heappush(abierta, (f_inicio, contador, inicio, None, g_inicio))

        cerrada = {}

        while abierta:
            _, _, pos_actual, padre, g_actual = heapq.heappop(abierta)

            if pos_actual in cerrada and cerrada[pos_actual][0] <= g_actual:
                continue

            cerrada[pos_actual] = (g_actual, padre)

            if pos_actual in destinos_posibles:
                camino = []
                while pos_actual is not None:
                    camino.append(pos_actual)
                    pos_actual = cerrada[pos_actual][1]
                return camino[::-1]

            movimientos = [
                (pos_actual[0] - 1, pos_actual[1]),
                (pos_actual[0] + 1, pos_actual[1]),
                (pos_actual[0], pos_actual[1] - 1),
                (pos_actual[0], pos_actual[1] + 1)
            ]

            for siguiente_pos in movimientos:
                if not self.almacen.es_valido(siguiente_pos):
                    continue

                g_siguiente = g_actual + 1

                if siguiente_pos in cerrada and cerrada[siguiente_pos][0] <= g_siguiente:
                    continue

                h_siguiente = min(self.heuristica(siguiente_pos, d) for d in destinos_posibles)
                f_siguiente = g_siguiente + h_siguiente

                contador += 1
                heapq.heappush(abierta, (f_siguiente, contador, siguiente_pos, pos_actual, g_siguiente))

        print("No se encontró un camino válido.")
        return []


class TempleSimulado:
    def __init__(self, almacen, orden, esquema_enfriamiento='exponencial',
                 temp_inicial=None, factor_exponencial=0.9995,
                 tasa_lineal=0.1, factor_logaritmico=0.01):
        self.almacen = almacen
        self.orden = orden
        self.estado_actual = orden[:]

        # Determinar temperatura inicial según tamaño de la orden si no se especifica
        if temp_inicial is None:
            self.temperatura = 100 * len(orden)  # Proporcional al tamaño
        else:
            self.temperatura = temp_inicial

        # Parámetros de enfriamiento
        self.esquema_enfriamiento = esquema_enfriamiento
        self.factor_exponencial = factor_exponencial
        self.tasa_lineal = tasa_lineal
        self.factor_logaritmico = factor_logaritmico

        self.agente_a_estrella = AgenteAEstrella(almacen)

        # Para recopilar estadísticas
        self.historial_costos = []
        self.mejor_costo_historial = []

    def enfriamiento(self, iteracion):
        """Aplica el esquema de enfriamiento seleccionado."""
        if self.esquema_enfriamiento == 'exponencial':
            self.temperatura *= self.factor_exponencial
        elif self.esquema_enfriamiento == 'lineal':
            self.temperatura = max(0.01, self.temperatura - self.tasa_lineal)
        elif self.esquema_enfriamiento == 'logaritmico':
            self.temperatura = self.temperatura / (1 + self.factor_logaritmico * math.log(1 + iteracion))

    def calcular_costo(self, orden=None):
        if orden is None:
            orden = self.estado_actual  # Si no se especifica, usa el estado actual

        costo_total = 0
        posicion_actual = self.almacen.carga_pos  # Inicia desde la zona de carga

        for producto in orden:
            camino = self.agente_a_estrella.encontrar_camino(posicion_actual, producto)

            if not camino:
                return float('inf')  # Retorna infinito si no hay camino válido

            distancia = len(camino) - 1
            costo_total += distancia

            # Actualizamos posición actual a la posición adyacente a la estantería
            posicion_actual = camino[-1]

        # Añadir el retorno a la zona de carga al final
        camino_final = self.agente_a_estrella.encontrar_camino(posicion_actual, None)
        if camino_final:
            costo_total += len(camino_final) - 1

        return costo_total

    def generar_vecino(self, estado: list) -> list:
        vecino = estado[:]
        tipo_movimiento = random.choice(["swap", "invertir", "mover"])

        if tipo_movimiento == "swap":
            # Intercambia dos productos al azar
            i, j = random.sample(range(len(vecino)), 2)
            vecino[i], vecino[j] = vecino[j], vecino[i]

        elif tipo_movimiento == "invertir":
            # Invierte un segmento aleatorio
            i, j = sorted(random.sample(range(len(vecino)), 2))
            vecino[i:j + 1] = reversed(vecino[i:j + 1])

        elif tipo_movimiento == "mover":
            # Mueve un producto a otra posición
            i, j = random.sample(range(len(vecino)), 2)
            item = vecino.pop(i)
            vecino.insert(j, item)

        return vecino

    def ejecutar(self, max_iteraciones=10000):
        random.seed(0)  # Para reproducibilidad
        costo_actual = self.calcular_costo(self.estado_actual)
        mejor_estado = self.estado_actual[:]
        mejor_costo = costo_actual

        self.historial_costos = []
        self.mejor_costo_historial = []

        for i in range(max_iteraciones):
            vecino = self.generar_vecino(self.estado_actual)
            costo_vecino = self.calcular_costo(vecino)

            delta_costo = costo_vecino - costo_actual

            # Comprobación segura para evitar el desbordamiento
            if delta_costo < 0:
                # Si el vecino es mejor, lo aceptamos siempre
                probabilidad = 1.0
            elif self.temperatura < 1e-10:
                # Si la temperatura es extremadamente baja, probabilidad cero
                probabilidad = 0.0
            else:
                # Cálculo normal con temperatura razonable
                probabilidad = math.exp(-delta_costo / self.temperatura)

            # Decidir si aceptamos el vecino
            if delta_costo < 0 or random.random() < probabilidad:
                self.estado_actual = vecino[:]
                costo_actual = costo_vecino

                # Actualizar mejor solución si corresponde
                if costo_actual < mejor_costo:
                    mejor_estado = self.estado_actual[:]
                    mejor_costo = costo_actual
                    # print(f"Nueva mejor solución encontrada en iteración {i}: {mejor_estado} con costo {mejor_costo}")

            # Registrar estadísticas
            self.historial_costos.append(costo_actual)
            self.mejor_costo_historial.append(mejor_costo)

            # Aplicar enfriamiento
            self.enfriamiento(i)

            # Criterio de parada
            if self.temperatura < 0.01:
                print(f"Temperatura mínima alcanzada en iteración {i}")
                break

        # Guarda el mejor estado y costo como atributos para acceso fácil
        self.mejor_estado = mejor_estado
        self.mejor_costo = mejor_costo
        self.iteraciones_realizadas = i + 1

        print(f"Temple simulado completado después de {self.iteraciones_realizadas} iteraciones")
        print(f"Mejor solución: {self.mejor_estado}")
        print(f"Costo óptimo: {self.mejor_costo}")

        # Devolver los tres valores que se esperan en ejecutar_experimento()
        return mejor_estado, mejor_costo, i + 1

    def obtener_camino_completo(self, mejor_secuencia=None):
        if mejor_secuencia is None:
            mejor_secuencia = self.mejor_estado if hasattr(self, 'mejor_estado') else self.estado_actual

        camino = [self.almacen.carga_pos]  # Inicia con la zona de carga
        for producto in mejor_secuencia:
            camino_parcial = self.agente_a_estrella.encontrar_camino(camino[-1], producto)
            if not camino_parcial:
                return []
            camino.extend(camino_parcial[1:])  # Añade el camino sin el inicio

        # Añadir retorno a la zona de carga al final
        camino_final = self.agente_a_estrella.encontrar_camino(camino[-1], None)
        if camino_final and len(camino_final) > 1:
            camino.extend(camino_final[1:])

        return camino

    def visualizar_camino(self, camino: list, destinos: list) -> None:
        if not camino:
            print("No hay camino para visualizar.")
            return

        visualizacion = np.zeros((self.almacen.filas, self.almacen.columnas))
        # Matriz para almacenar los números de pasos
        numeros_pasos = np.full((self.almacen.filas, self.almacen.columnas), "", dtype=object)

        # Marca las estanterías
        for producto, pos in self.almacen.estanterias.items():
            visualizacion[pos] = 1

        # Marca la zona de carga
        visualizacion[self.almacen.carga_pos] = 5

        # Definir colores distintos para cada tramo
        colores = [2, 3, 4, 6, 7, 8]  # Diferentes valores para los colores
        color_idx = 0

        # Diccionario para rastrear productos visitados
        productos_visitados = {}

        # Obtiene posiciones adyacentes a productos
        posiciones_productos = {}
        for producto in destinos:
            pos_adyacentes = self.almacen.get_posiciones_adyacentes(producto)
            if pos_adyacentes:
                posiciones_productos[producto] = pos_adyacentes[0]

        # Marcar el camino con distintos colores por tramo y numerar cada paso
        for i in range(len(camino)):
            pos = camino[i]
            fila, col = pos

            # Si ya pasamos por esta casilla, agregamos el nuevo número al existente
            if numeros_pasos[fila, col]:
                numeros_pasos[fila, col] += f",{i}"
            else:
                numeros_pasos[fila, col] = f"{i}"

            # Asignar color correspondiente al tramo actual
            if i > 0:  # El punto inicial (zona de carga) tiene su propio color
                visualizacion[pos] = colores[color_idx]

            # Detectar si llegamos a una posición adyacente a un producto
            for producto, pos_adyacente in posiciones_productos.items():
                if pos == pos_adyacente and producto not in productos_visitados:
                    productos_visitados[producto] = i
                    if i < len(camino) - 1:  # No cambiar color en el último paso
                        color_idx = (color_idx + 1) % len(colores)
                    break

        # Gráfica
        plt.figure(figsize=(14, 12))

        # Color map para la visualización
        cmap = plt.cm.colors.ListedColormap(
            ['white', 'lightgray', 'blue', 'green', 'red', 'purple', 'orange', 'yellow'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(visualizacion, cmap=cmap, norm=norm)

        # Numerar las estanterías
        for producto, pos in self.almacen.estanterias.items():
            plt.text(pos[1], pos[0], str(producto), ha='center', va='center', color='black', fontsize=8,
                     fontweight='bold')

        # Mostrar números de paso en cada casilla
        for f in range(self.almacen.filas):
            for c in range(self.almacen.columnas):
                # Si es una casilla por la que pasó el camino
                if numeros_pasos[f, c]:
                    # Si la celda contiene muchos pasos, mostramos sólo el primero y el último
                    pasos = numeros_pasos[f, c].split(',')
                    if len(pasos) > 3:
                        texto_paso = f"{pasos[0]}..{pasos[-1]}"
                    else:
                        texto_paso = numeros_pasos[f, c]

                    # Elegir color de texto según color de fondo
                    color_texto = 'white' if visualizacion[f, c] in [2, 4, 6, 8] else 'black'

                    # Mostrar el número de paso
                    plt.text(c, f, texto_paso, ha='center', va='center',
                             color=color_texto, fontsize=7, fontweight='bold')

                # Para estanterías y casillas vacías, mostrar coordenadas
                elif visualizacion[f, c] != 1 and visualizacion[f, c] != 5:
                    plt.text(c, f, f"({f},{c})", ha='center', va='center', color='black', fontsize=7, alpha=0.5)

        # Añadir flechas para mostrar la dirección del recorrido
        for i in range(1, len(camino)):
            y_prev, x_prev = camino[i - 1]
            y_curr, x_curr = camino[i]

            # Calcular la dirección de la flecha
            dx = (x_curr - x_prev) * 0.4
            dy = (y_curr - y_prev) * 0.4

            # Dibujar flecha desde el punto medio de las celdas
            plt.arrow(x_prev, y_prev, dx, dy, head_width=0.2, head_length=0.2,
                      fc='black', ec='black', alpha=0.7, length_includes_head=True)

        # Etiqueta zona de carga
        plt.text(self.almacen.carga_pos[1], self.almacen.carga_pos[0], 'C', ha='center', va='center', color='black',
                 fontweight='bold')

        # Agregar cuadrícula
        plt.grid(color='black', linestyle='-', linewidth=0.5)

        # Configuración de ejes
        plt.xticks(np.arange(-0.5, self.almacen.columnas, 1), [])
        plt.yticks(np.arange(-0.5, self.almacen.filas, 1), [])
        plt.xlim(-0.5, self.almacen.columnas - 0.5)
        plt.ylim(-0.5, self.almacen.filas - 0.5)

        # Añadir leyenda para los productos recogidos
        leyenda_texto = []
        for producto in destinos:
            paso = productos_visitados.get(producto, "N/A")
            leyenda_texto.append(f"Producto {producto}: paso {paso}")

        # Añadir leyenda para los colores
        handles = [Patch(color='blue', label='Tramo 1'),
                   Patch(color='green', label='Tramo 2'),
                   Patch(color='red', label='Tramo 3'),
                   Patch(color='purple', label='Tramo 4'),
                   Patch(color='orange', label='Tramo 5'),
                   Patch(color='yellow', label='Tramo 6')]

        plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1))

        # Añadir una tabla con los productos recogidos
        tabla_ax = plt.axes([0.65, 0.05, 0.3, 0.3])
        tabla_ax.axis('off')
        tabla = tabla_ax.table(
            cellText=[[f"Producto {p}", f"Paso {productos_visitados.get(p, 'N/A')}"] for p in destinos],
            colLabels=["Producto", "Paso"],
            loc='center',
            cellLoc='center'
        )
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        tabla.scale(1.2, 1.5)

        plt.title(f"Recorrido para recoger productos: {destinos}\nLongitud total: {len(camino) - 1} pasos", fontsize=12)
        plt.tight_layout()
        plt.show()


# Clase para experimentación con temple simulado
class ExperimentadorTS:
    def __init__(self, almacen):
        self.almacen = almacen
        self.resultados = {}

    def generar_orden_aleatoria(self, tamaño):
        """Genera una orden aleatoria de productos de tamaño especificado."""
        productos_disponibles = list(self.almacen.estanterias.keys())
        return random.sample(productos_disponibles, min(tamaño, len(productos_disponibles)))

    def ejecutar_experimento(self, tamaño_orden, num_ordenes, max_iteraciones=10000,
                             esquemas_enfriamiento=['exponencial', 'lineal', 'logaritmico']):
        """Ejecuta experimentos con ordenes del mismo tamaño."""
        self.resultados[tamaño_orden] = {esquema: {
            'costos_finales': [],
            'historial_costos': [],
            'iteraciones': [],
            'tiempos': []
        } for esquema in esquemas_enfriamiento}

        for esquema in esquemas_enfriamiento:
            print(f"Ejecutando experimentos con esquema {esquema}...")
            for i in range(num_ordenes):
                if i % 10 == 0:
                    print(f"  Orden {i + 1}/{num_ordenes}")

                orden = self.generar_orden_aleatoria(tamaño_orden)

                import time
                inicio = time.time()

                ts = TempleSimulado(self.almacen, orden, esquema_enfriamiento=esquema)
                mejor_estado, mejor_costo, iteraciones = ts.ejecutar(max_iteraciones)

                fin = time.time()

                self.resultados[tamaño_orden][esquema]['costos_finales'].append(mejor_costo)
                self.resultados[tamaño_orden][esquema]['historial_costos'].append(ts.mejor_costo_historial)
                self.resultados[tamaño_orden][esquema]['iteraciones'].append(iteraciones)
                self.resultados[tamaño_orden][esquema]['tiempos'].append(fin - inicio)

    def analizar_resultados(self, tamaño_orden):
        """Analiza y muestra los resultados del experimento."""
        if tamaño_orden not in self.resultados:
            print(f"No hay resultados para tamaño {tamaño_orden}")
            return

        for esquema, datos in self.resultados[tamaño_orden].items():
            costos = datos['costos_finales']
            iteraciones = datos['iteraciones']
            tiempos = datos['tiempos']

            print(f"Esquema: {esquema}")
            print(f"  Costo promedio: {sum(costos) / len(costos):.2f}")
            print(f"  Desviación estándar: {np.std(costos):.2f}")
            print(f"  Iteraciones promedio: {sum(iteraciones) / len(iteraciones):.2f}")
            print(f"  Tiempo promedio: {sum(tiempos) / len(tiempos):.4f} segundos")
            print()

    def graficar_convergencia(self, tamaño_orden):
        """Grafica la convergencia de los distintos esquemas de enfriamiento."""
        if tamaño_orden not in self.resultados:
            print(f"No hay resultados para tamaño {tamaño_orden}")
            return

        plt.figure(figsize=(12, 6))

        for esquema, datos in self.resultados[tamaño_orden].items():
            # Promediamos los historiales
            max_len = max(len(hist) for hist in datos['historial_costos'])
            historiales_normalizados = []

            for hist in datos['historial_costos']:
                # Extender historiales más cortos
                if len(hist) < max_len:
                    extendido = hist + [hist[-1]] * (max_len - len(hist))
                    historiales_normalizados.append(extendido)
                else:
                    historiales_normalizados.append(hist)

            # Calcular promedio
            promedio = np.mean(historiales_normalizados, axis=0)

            # Graficar
            plt.plot(promedio, label=f"{esquema}")

        plt.title(f"Convergencia promedio para órdenes de tamaño {tamaño_orden}")
        plt.xlabel("Iteraciones")
        plt.ylabel("Mejor costo")
        plt.legend()
        plt.grid(True)
        plt.show()

    def graficar_comparacion_tamaños(self, tamaños, esquema='exponencial'):
        """Compara la convergencia para diferentes tamaños de orden."""
        plt.figure(figsize=(12, 6))

        for tamaño in tamaños:
            if tamaño not in self.resultados:
                print(f"No hay resultados para tamaño {tamaño}")
                continue

            if esquema not in self.resultados[tamaño]:
                print(f"No hay resultados para el esquema {esquema} en tamaño {tamaño}")
                continue

            datos = self.resultados[tamaño][esquema]

            # Promediamos los historiales
            max_len = max(len(hist) for hist in datos['historial_costos'])
            historiales_normalizados = []

            for hist in datos['historial_costos']:
                if len(hist) < max_len:
                    extendido = hist + [hist[-1]] * (max_len - len(hist))
                    historiales_normalizados.append(extendido)
                else:
                    historiales_normalizados.append(hist)

            # Calcular promedio
            promedio = np.mean(historiales_normalizados, axis=0)

            # Graficar
            plt.plot(promedio, label=f"Tamaño {tamaño}")

        plt.title(f"Comparación de convergencia por tamaño de orden ({esquema})")
        plt.xlabel("Iteraciones")
        plt.ylabel("Mejor costo")
        plt.legend()
        plt.grid(True)
        plt.show()

    def guardar_resultados(self, archivo):
        """Guarda los resultados de los experimentos en un archivo."""


# Punto de entrada principal del programa
if __name__ == "__main__":
    # Imprime el menú al inicio
    print("\n======= OPTIMIZADOR DE RUTAS CON TEMPLE SIMULADO =======")
    print("Modos disponibles:")
    print("1: Temple Simulado simple")
    print("2: Experimentación y comparación de esquemas")
    modo = input("Seleccione modo (1 o 2): ")

    # Crea el almacén en cualquier caso
    almacen = Almacen()

    if modo == "1":
        # Ejecución simple
        print("\nIngrese la orden inicial (lista de productos a recoger)")
        orden_input = input("Números de productos separados por comas: ")
        orden_inicial = [int(producto) for producto in orden_input.split(',')]

        print("\nElegir esquema de enfriamiento:")
        print("1: Exponencial (clásico)")
        print("2: Lineal")
        print("3: Logarítmico")
        esquema_opt = input("Seleccione esquema (1-3): ")

        esquema = "exponencial"  # Valor por defecto
        if esquema_opt == "2":
            esquema = "lineal"
        elif esquema_opt == "3":
            esquema = "logaritmico"

        print(f"\nEjecutando temple simulado con esquema {esquema}...")
        temple = TempleSimulado(almacen, orden_inicial, esquema_enfriamiento=esquema)
        orden_optimo, mejor_costo, iteraciones = temple.ejecutar()

        print("\nResultados:")
        print("Orden inicial:", orden_inicial)
        print("Orden óptimo:", orden_optimo)
        print("Distancia total óptima:", mejor_costo)
        print("Iteraciones realizadas:", iteraciones)
        print("Esquema de enfriamiento:", esquema)

        # Visualizar camino
        print("\nGenerando visualización del camino óptimo...")
        camino_completo = temple.obtener_camino_completo(orden_optimo)
        temple.visualizar_camino(camino_completo, orden_optimo)

    elif modo == "2":
        # Modo experimentación
        experimentador = ExperimentadorTS(almacen)

        # Parámetros de experimentación
        print("\nConfiguración del experimento:")
        tamaño = int(input("Ingrese tamaño de las órdenes a generar: "))
        num_ordenes = int(input("Ingrese número de órdenes a probar: "))

        print("\nEjecutando experimentos...")
        experimentador.ejecutar_experimento(tamaño_orden=tamaño, num_ordenes=num_ordenes)

        print("\nAnálisis de resultados:")
        experimentador.analizar_resultados(tamaño_orden=tamaño)

        # Visualizar resultados
        print("\nGenerando gráfica de convergencia...")
        experimentador.graficar_convergencia(tamaño_orden=tamaño)
    else:
        print("Modo no válido. Ejecute el programa nuevamente y seleccione 1 o 2.")

