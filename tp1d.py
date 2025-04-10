import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import List, Dict, Tuple, Any
import heapq
from collections import Counter
import seaborn as sns

class Almacen:
    def __init__(self, configuracion=None):
        self.filas = 11
        self.columnas = 13
        self.layout = np.full((self.filas, self.columnas), ' ', dtype=object)
        
        self.posiciones_estanterias = {
            1:  (9, 2),  2:  (9, 3),
            3:  (8, 2),  4:  (8, 3),
            5:  (7, 2),  6:  (7, 3),
            7:  (6, 2),  8:  (6, 3),
            9:  (9, 6),  10: (9, 7),
            11: (8, 6),  12: (8, 7),
            13: (7, 6),  14: (7, 7),
            15: (6, 6),  16: (6, 7),
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
        
        if configuracion is None:
            self.estanterias = {i: i for i in range(1, 49)}
        else:
            self.estanterias = {i: configuracion[i-1] for i in range(1, 49)}
        
        self.ubicacion_productos = {}
        for estanteria, producto in self.estanterias.items():
            self.ubicacion_productos[producto] = self.posiciones_estanterias[estanteria]
        
        for estanteria, producto in self.estanterias.items():
            fila, col = self.posiciones_estanterias[estanteria]
            self.layout[fila, col] = str(producto)
        
        self.carga_pos = (5, 0)
        self.layout[self.carga_pos] = 'C'
        
        self.distancias_estanterias = {}
        for estanteria, (fila, col) in self.posiciones_estanterias.items():
            dist = abs(fila - self.carga_pos[0]) + abs(col - self.carga_pos[1])
            self.distancias_estanterias[estanteria] = dist
    
    def es_valido(self, pos: tuple) -> bool:
        fila, col = pos
        if fila < 0 or fila >= self.filas or col < 0 or col >= self.columnas:
            return False
        if self.layout[fila, col] not in [' ', 'C']:
            return False
        return True
    
    def get_posiciones_adyacentes(self, producto: int) -> list:
        if producto not in self.ubicacion_productos:
            return []
        fila, col = self.ubicacion_productos[producto]
        adyacentes = [(fila, col-1), (fila, col+1)]
        return [pos for pos in adyacentes if self.es_valido(pos)]


class AgenteAEstrella:
    def __init__(self, almacen: Almacen):
        self.almacen = almacen

    def heuristica(self, pos: Tuple[int, int], destino: Tuple[int, int]) -> float:
        return abs(pos[0] - destino[0]) + abs(pos[1] - destino[1])

    def encontrar_camino(self, inicio: Tuple[int, int], destino_producto: int = None) -> List[Tuple[int, int]]:
        if inicio is None:
            inicio = self.almacen.carga_pos

        if destino_producto is None:
            destino = self.almacen.carga_pos
            destinos_posibles = [destino]
        else:
            destinos_posibles = self.almacen.get_posiciones_adyacentes(destino_producto)
            if not destinos_posibles:
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
                (pos_actual[0]-1, pos_actual[1]),
                (pos_actual[0]+1, pos_actual[1]),
                (pos_actual[0], pos_actual[1]-1),
                (pos_actual[0], pos_actual[1]+1)
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

        return []


class OptimizadorRecocido:
    def __init__(self, almacen: Almacen, orden: List[int], factor_logaritmico=0.008):
        self.almacen = almacen
        self.orden = orden
        self.estado_actual = orden[:]
        self.temperatura = 100 * len(orden)
        self.factor_logaritmico = factor_logaritmico
        self.agente_a_estrella = AgenteAEstrella(almacen)
    
    def enfriamiento(self, iteracion):
        self.temperatura = self.temperatura / (1 + self.factor_logaritmico * np.log(1 + iteracion))
    
    def calcular_costo(self, orden=None):
        if orden is None:
            orden = self.estado_actual

        costo_total = 0
        posicion_actual = self.almacen.carga_pos

        for producto in orden:
            camino = self.agente_a_estrella.encontrar_camino(posicion_actual, producto)

            if not camino:
                return float('inf')

            distancia = len(camino) - 1
            costo_total += distancia
            posicion_actual = camino[-1]
        
        camino_final = self.agente_a_estrella.encontrar_camino(posicion_actual, None)
        if camino_final:
            costo_total += len(camino_final) - 1

        return costo_total

    def generar_vecino(self, estado: list) -> list:
        vecino = estado[:]
        i, j = random.sample(range(len(vecino)), 2)
        vecino[i], vecino[j] = vecino[j], vecino[i]
        return vecino

    def ejecutar(self, max_iteraciones=1000):
        costo_actual = self.calcular_costo(self.estado_actual)
        mejor_estado = self.estado_actual[:]
        mejor_costo = costo_actual
        
        for i in range(max_iteraciones):
            vecino = self.generar_vecino(self.estado_actual)
            costo_vecino = self.calcular_costo(vecino)
            
            delta_costo = costo_vecino - costo_actual
            
            if delta_costo < 0:
                probabilidad = 1.0
            elif self.temperatura < 1e-10:
                probabilidad = 0.0
            else:
                probabilidad = np.exp(-delta_costo / self.temperatura)
            
            if delta_costo < 0 or random.random() < probabilidad:
                self.estado_actual = vecino[:]
                costo_actual = costo_vecino
                
                if costo_actual < mejor_costo:
                    mejor_estado = self.estado_actual[:]
                    mejor_costo = costo_actual
            
            self.enfriamiento(i)
            
            if self.temperatura < 0.01:
                break
                
        return mejor_estado, mejor_costo


class AlgoritmoGenetico:
    def __init__(self, historico_ordenes: List[List[int]], num_productos=48, 
                 tam_poblacion=50, prob_cruce=0.8, prob_mutacion=0.2, 
                 num_elitismo=2, max_generaciones=100):
        self.historico_ordenes = historico_ordenes
        self.num_productos = num_productos
        self.tam_poblacion = tam_poblacion
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.num_elitismo = num_elitismo
        self.max_generaciones = max_generaciones
        self.frecuencia_productos = self._contar_frecuencia_productos()
        self.poblacion = self._inicializar_poblacion()
        self.mejor_individuo = None
        self.mejor_fitness = float('inf')
        self.historial_mejor_fitness = []
        self.historial_fitness_promedio = []
        almacen_temp = Almacen()
        self.distancias_estanterias = almacen_temp.distancias_estanterias
        self._cache_fitness = {}
    
    def _contar_frecuencia_productos(self) -> Dict[int, int]:
        contador = Counter()
        for orden in self.historico_ordenes:
            contador.update(orden)
        return dict(contador)
    
    def _inicializar_poblacion(self) -> List[List[int]]:
        poblacion = []
        
        # Inicializar almacén temporal para obtener distancias
        almacen_temp = Almacen()
        distancias = almacen_temp.distancias_estanterias
        
        
        for _ in range(int(self.tam_poblacion * 0.6)): 
            individuo = list(range(1, self.num_productos + 1))
            random.shuffle(individuo)
            poblacion.append(individuo)
        
        
        for _ in range(int(self.tam_poblacion * 0.3)):
            productos_ordenados = sorted(
                range(1, self.num_productos + 1),
                key=lambda p: self.frecuencia_productos.get(p, 0),
                reverse=True
            )
            
            # Aplicar un poco de aleatorización
            for i in range(len(productos_ordenados)):
                if random.random() < 0.2:
                    j = random.randint(0, len(productos_ordenados) - 1)
                    productos_ordenados[i], productos_ordenados[j] = productos_ordenados[j], productos_ordenados[i]
            
            poblacion.append(productos_ordenados)
        
        
        for _ in range(int(self.tam_poblacion * 0.1)):
            # Ordenar estanterías por distancia a la carga
            estanterias_ordenadas = sorted(
                range(1, self.num_productos + 1),
                key=lambda e: distancias.get(e, float('inf'))
            )
            
            # Ordenar productos por frecuencia
            productos_ordenados = sorted(
                range(1, self.num_productos + 1),
                key=lambda p: self.frecuencia_productos.get(p, 0),
                reverse=True
            )
            
            # Crear configuración donde productos frecuentes van en estanterías cercanas
            configuracion = [0] * self.num_productos
            for i, estanteria in enumerate(estanterias_ordenadas):
                if i < len(productos_ordenados):
                    configuracion[estanteria-1] = productos_ordenados[i]
            
            # Aplicar un poco de aleatorización
            for i in range(len(configuracion)):
                if random.random() < 0.1:
                    j = random.randint(0, len(configuracion) - 1)
                    configuracion[i], configuracion[j] = configuracion[j], configuracion[i]
                    
            poblacion.append(configuracion)
        
        return poblacion
    
    def calcular_fitness(self, individuo: List[int]) -> float:
        individuo_tupla = tuple(individuo)
        
        if individuo_tupla in self._cache_fitness:
            return self._cache_fitness[individuo_tupla]
        
        almacen = Almacen(configuracion=individuo)
        
        penalizacion_distancia = 0
        total_frecuencia = sum(self.frecuencia_productos.values())
        
        for producto, freq in self.frecuencia_productos.items():
            for estanteria, prod in almacen.estanterias.items():
                if prod == producto:
                    # Usar las distancias calculadas desde (5,0)
                    distancia = almacen.distancias_estanterias[estanteria]
                    freq_norm = freq / total_frecuencia
                    # Mayor penalización para productos frecuentes lejos de la carga
                    penalizacion_distancia += distancia * freq_norm
                    break
        
        ordenes_a_evaluar = self.historico_ordenes
        if len(self.historico_ordenes) > 30:
            ordenes_a_evaluar = random.sample(
                self.historico_ordenes, 
                min(30, len(self.historico_ordenes))
            )
        
        costo_picking_total = 0
        num_ordenes = 0
        
        for orden in ordenes_a_evaluar:
            if len(orden) > 30:
                orden_muestreada = random.sample(orden, 30)
            else:
                orden_muestreada = orden
            
            num_ordenes += 1
            
            optimizador = OptimizadorRecocido(almacen, orden_muestreada)
            _, costo = optimizador.ejecutar(max_iteraciones=100)
            
            costo_picking_total += costo
        
        costo_picking_promedio = costo_picking_total / max(1, num_ordenes)
        
        fitness = 0.35 * penalizacion_distancia + 0.65 * costo_picking_promedio
        
        self._cache_fitness[individuo_tupla] = fitness
        
        return fitness
    
    def seleccion_torneo(self, tam_torneo: int = 3) -> List[int]:
        candidatos = random.sample(range(self.tam_poblacion), tam_torneo)
        fitness_candidatos = [self.calcular_fitness(self.poblacion[i]) for i in candidatos]
        return self.poblacion[candidatos[np.argmin(fitness_candidatos)]]
    
    def order_crossover(self, padre1: List[int], padre2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Implementa el operador Order Crossover (OX) para problemas de permutación.
        
        Args:
            padre1: Una permutación (lista de enteros)
            padre2: Una permutación (lista de enteros)
            
        Returns:
            Una tupla con dos hijos que son permutaciones válidas
        """
        n = len(padre1)
        
        # Seleccionar dos puntos de corte aleatorios
        punto1, punto2 = sorted(random.sample(range(n), 2))
        
        # Creación del primer hijo
        hijo1 = [None] * n
        
        # Paso 1: Copiar el segmento del primer padre
        for i in range(punto1, punto2 + 1):
            hijo1[i] = padre1[i]
        
        # Paso 2: Copiar los elementos restantes en el orden del segundo padre
        elementos_restantes_p2 = [item for item in padre2 if item not in hijo1[punto1:punto2+1]]
        
        idx = (punto2 + 1) % n
        for item in elementos_restantes_p2:
            while hijo1[idx] is not None:
                idx = (idx + 1) % n
            hijo1[idx] = item
            idx = (idx + 1) % n
        
        # Creación del segundo hijo (proceso inverso)
        hijo2 = [None] * n
        
        # Paso 1: Copiar el segmento del segundo padre
        for i in range(punto1, punto2 + 1):
            hijo2[i] = padre2[i]
        
        # Paso 2: Copiar los elementos restantes en el orden del primer padre
        elementos_restantes_p1 = [item for item in padre1 if item not in hijo2[punto1:punto2+1]]
        
        idx = (punto2 + 1) % n
        for item in elementos_restantes_p1:
            while hijo2[idx] is not None:
                idx = (idx + 1) % n
            hijo2[idx] = item
            idx = (idx + 1) % n
        
        return hijo1, hijo2
    
    def mutacion_swap(self, individuo: List[int]) -> List[int]:
        resultado = individuo[:]
        i, j = random.sample(range(len(resultado)), 2)
        resultado[i], resultado[j] = resultado[j], resultado[i]
        return resultado
    
    def ejecutar(self) -> Dict[str, Any]:
        tiempo_inicio = time.time()
        
        fitness_poblacion = [self.calcular_fitness(ind) for ind in self.poblacion]
        
        mejor_idx = np.argmin(fitness_poblacion)
        self.mejor_individuo = self.poblacion[mejor_idx][:]
        self.mejor_fitness = fitness_poblacion[mejor_idx]
        
        self.historial_mejor_fitness.append(self.mejor_fitness)
        self.historial_fitness_promedio.append(np.mean(fitness_poblacion))
        
        print(f"Generación 0: Mejor fitness = {self.mejor_fitness:.2f}, Fitness promedio = {np.mean(fitness_poblacion):.2f}")
        
        generaciones_sin_mejora = 0
        umbral_convergencia = 40
        
        for generacion in range(1, self.max_generaciones + 1):
            tiempo_gen_inicio = time.time()
            
            fitness_con_indices = [(f, i) for i, f in enumerate(fitness_poblacion)]
            fitness_con_indices.sort()
            
            indices_elite = [i for _, i in fitness_con_indices[:self.num_elitismo]]
            nueva_poblacion = [self.poblacion[i][:] for i in indices_elite]
            
            while len(nueva_poblacion) < self.tam_poblacion:
                if random.random() < self.prob_cruce and len(nueva_poblacion) + 2 <= self.tam_poblacion:
                    padre1 = self.seleccion_torneo()
                    padre2 = self.seleccion_torneo()
                    
                    # Usar Order Crossover
                    hijo1, hijo2 = self.order_crossover(padre1, padre2)
                    
                    if random.random() < self.prob_mutacion:
                        hijo1 = self.mutacion_swap(hijo1)
                    if random.random() < self.prob_mutacion:
                        hijo2 = self.mutacion_swap(hijo2)
                    
                    nueva_poblacion.extend([hijo1, hijo2])
                else:
                    individuo = self.seleccion_torneo()
                    if random.random() < self.prob_mutacion:
                        individuo = self.mutacion_swap(individuo)
                    nueva_poblacion.append(individuo)
            
            self.poblacion = nueva_poblacion[:self.tam_poblacion]
            
            fitness_poblacion = [self.calcular_fitness(ind) for ind in self.poblacion]
            
            mejor_generacion_idx = np.argmin(fitness_poblacion)
            mejor_fitness_actual = fitness_poblacion[mejor_generacion_idx]
            
            if mejor_fitness_actual < self.mejor_fitness:
                self.mejor_individuo = self.poblacion[mejor_generacion_idx][:]
                self.mejor_fitness = mejor_fitness_actual
                generaciones_sin_mejora = 0
                print(f"Generación {generacion}: Nuevo mejor fitness = {self.mejor_fitness:.2f}")
            else:
                generaciones_sin_mejora += 1
            
            self.historial_mejor_fitness.append(self.mejor_fitness)
            self.historial_fitness_promedio.append(np.mean(fitness_poblacion))
            
            tiempo_gen = time.time() - tiempo_gen_inicio
            
            if generacion % 5 == 0:
                print(f"Generación {generacion}: Mejor fitness = {self.mejor_fitness:.2f}, "
                      f"Fitness promedio = {np.mean(fitness_poblacion):.2f}, "
                      f"Tiempo: {tiempo_gen:.2f}s")
            
            if generaciones_sin_mejora >= umbral_convergencia:
                print(f"Algoritmo convergido después de {generacion} generaciones sin mejora.")
                break
            
            tiempo_actual = time.time() - tiempo_inicio
            if tiempo_actual >1500:
                print("Tiempo máximo alcanzado. Deteniendo algoritmo.")
                break
        
        tiempo_total = time.time() - tiempo_inicio
        print(f"Algoritmo genético completado en {tiempo_total:.2f} segundos")
        print(f"Mejor fitness final: {self.mejor_fitness:.2f}")
        
        return {
            "mejor_individuo": self.mejor_individuo,
            "mejor_fitness": self.mejor_fitness,
            "historial_mejor": self.historial_mejor_fitness,
            "historial_promedio": self.historial_fitness_promedio,
            "tiempo_total": tiempo_total
        }

    def visualizar_evolucion(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.historial_mejor_fitness, label="Mejor fitness")
        plt.plot(self.historial_fitness_promedio, label="Fitness promedio")
        plt.xlabel("Generación")
        plt.ylabel("Fitness (costo promedio)")
        plt.title("Evolución del fitness durante la ejecución del algoritmo genético")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def visualizar_almacen_optimizado(self):
        if self.mejor_individuo is None:
            print("No hay una solución optimizada para visualizar")
            return
        
        almacen = Almacen(configuracion=self.mejor_individuo)
        
        mapa_calor = np.zeros((almacen.filas, almacen.columnas))
        
        max_frecuencia = max(self.frecuencia_productos.values()) if self.frecuencia_productos else 1
        
        for estanteria, producto in almacen.estanterias.items():
            fila, col = almacen.posiciones_estanterias[estanteria]
            frecuencia = self.frecuencia_productos.get(producto, 0)
            mapa_calor[fila, col] = frecuencia / max_frecuencia
        
        plt.figure(figsize=(14, 12))
        
        cmap = plt.cm.get_cmap('coolwarm')
        plt.imshow(mapa_calor, cmap=cmap, interpolation='nearest')
        
        for estanteria, producto in almacen.estanterias.items():
            fila, col = almacen.posiciones_estanterias[estanteria]
            frecuencia = self.frecuencia_productos.get(producto, 0)
            plt.text(col, fila, f"P:{producto}\nF:{frecuencia}", 
                    ha='center', va='center', 
                    color='black' if mapa_calor[fila, col] < 0.6 else 'white', 
                    fontsize=8, fontweight='bold')
        
        plt.text(almacen.carga_pos[1], almacen.carga_pos[0], 'CARGA', 
                ha='center', va='center', color='white', 
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle="round", facecolor='blue', alpha=0.7))
        
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(almacen.columnas), np.arange(almacen.columnas))
        plt.yticks(np.arange(almacen.filas), np.arange(almacen.filas))
        
        cbar = plt.colorbar(label='Frecuencia normalizada')
        cbar.set_label('Frecuencia normalizada', rotation=270, labelpad=20)
        
        plt.title('Configuración Optimizada del Almacén\nMapa de calor por frecuencia de productos')
        plt.tight_layout()
        plt.show()


def cargar_ordenes_historicas(archivo):
    try:
        with open(archivo, 'r') as f:
            ordenes = []
            for linea in f:
                items = [item.strip() for item in linea.strip().split(',')]
                orden = [int(item) for item in items if item]
                if orden:
                    ordenes.append(orden)
        
        print(f"Se leyeron {len(ordenes)} órdenes correctamente.")
        return ordenes
    except Exception as e:
        print(f"Error al cargar las órdenes: {e}")
        return [
            [5, 10, 15, 20],
            [1, 3, 5, 7, 9],
            [2, 4, 6, 8, 10],
            [11, 22, 33, 44],
            [12, 24, 36, 48]
        ]


def evaluar_configuracion(configuracion, historico_ordenes, max_iteraciones_recocido=100):
    almacen = Almacen(configuracion=configuracion)
    
    costos_ordenes = []
    
    for i, orden in enumerate(historico_ordenes):
        optimizador = OptimizadorRecocido(almacen, orden)
        _, costo = optimizador.ejecutar(max_iteraciones=max_iteraciones_recocido)
        costos_ordenes.append(costo)
        
        if (i + 1) % 10 == 0:
            print(f"  Evaluadas {i + 1}/{len(historico_ordenes)} órdenes...")
    
    return np.mean(costos_ordenes)

def crear_configuracion_experta(frecuencia_productos):
    """
    Crea una configuración donde los productos más frecuentes están
    cerca de la zona de carga (medio-izquierda)
    
    Args:
        frecuencia_productos: Diccionario con las frecuencias de cada producto
    
    Returns:
        Lista de 48 elementos con la configuración propuesta
    """
    # Ordenar productos por frecuencia (descendente)
    productos_ordenados_por_frecuencia = sorted(
        range(1, 49), 
        key=lambda p: frecuencia_productos.get(p, 0), 
        reverse=True
    )
    
    # Definir el orden de las estanterías basado en la cercanía a la zona de carga (5,0)
    # El almacén tiene 48 estanterías en 3 pasillos:
    # - Pasillo izquierdo (más cercano a la carga): Estanterías 25-48
    # - Pasillo central: Estanterías 9-24
    # - Pasillo derecho (más alejado): Estanterías 1-8 
    
    # Vamos a ordenar las posiciones de estanterías por cercanía a (5,0)
    almacen_temp = Almacen()
    
    # Calcular distancia de cada estantería a la zona de carga (5,0)
    distancias_a_carga = {}
    for estanteria, (fila, col) in almacen_temp.posiciones_estanterias.items():
        dist = abs(fila - 5) + abs(col - 0)  # Distancia Manhattan a (5,0)
        distancias_a_carga[estanteria] = dist
    
    # Ordenar estanterías por cercanía a la zona de carga
    estanterias_ordenadas_por_cercania = sorted(
        range(1, 49), 
        key=lambda e: distancias_a_carga[e]
    )
    
    # Asignar productos a estanterías: los más frecuentes en las más cercanas
    configuracion_experta = [0] * 48  # Lista que vamos a llenar
    
    for i, estanteria in enumerate(estanterias_ordenadas_por_cercania):
        if i < len(productos_ordenados_por_frecuencia):
            producto = productos_ordenados_por_frecuencia[i]
            configuracion_experta[estanteria-1] = producto
    
    return configuracion_experta

# Función para comparar las tres configuraciones
def comparar_configuraciones(historico_ordenes, resultado_ag):
    """
    Compara tres configuraciones: la por defecto, la experta y la del algoritmo genético
    
    Args:
        historico_ordenes: Lista de órdenes históricas
        resultado_ag: Resultado del algoritmo genético
    """
    print("\nComparando las tres configuraciones...")
    
    # Configuración por defecto (productos secuenciales)
    config_default = list(range(1, 49))
    
    # Calcular frecuencias de productos
    contador_productos = Counter()
    for orden in historico_ordenes:
        contador_productos.update(orden)
    
    # Crear configuración humana experta basada en frecuencias
    config_experta = crear_configuracion_experta(contador_productos)
    
    # Configuración optimizada por el algoritmo genético
    config_ag = resultado_ag['mejor_individuo']
    
    # Evaluar las tres configuraciones
    print("Evaluando configuración por defecto...")
    almacen_default = Almacen(configuracion=config_default)
    costo_total_default = 0
    
    numero_ordenes_evaluar = min(50, len(historico_ordenes))
    ordenes_a_evaluar = historico_ordenes[:numero_ordenes_evaluar]
    
    for orden in ordenes_a_evaluar:
        optimizador = OptimizadorRecocido(almacen_default, orden)
        _, costo = optimizador.ejecutar(max_iteraciones=100)
        costo_total_default += costo
    
    fitness_default = costo_total_default / numero_ordenes_evaluar
    
    print("Evaluando configuración humana experta...")
    almacen_experto = Almacen(configuracion=config_experta)
    costo_total_experto = 0
    
    for orden in ordenes_a_evaluar:
        optimizador = OptimizadorRecocido(almacen_experto, orden)
        _, costo = optimizador.ejecutar(max_iteraciones=100)
        costo_total_experto += costo
    
    fitness_experto = costo_total_experto / numero_ordenes_evaluar
    
    print("Evaluando configuración optimizada por AG...")
    almacen_optimizado = Almacen(configuracion=config_ag)
    costo_total_optimizado = 0
    
    for orden in ordenes_a_evaluar:
        optimizador = OptimizadorRecocido(almacen_optimizado, orden)
        _, costo = optimizador.ejecutar(max_iteraciones=100)
        costo_total_optimizado += costo
    
    fitness_optimizado = costo_total_optimizado / numero_ordenes_evaluar
    
    # Calcular mejoras porcentuales
    mejora_ag_vs_default = ((fitness_default - fitness_optimizado) / fitness_default) * 100
    mejora_ag_vs_experto = ((fitness_experto - fitness_optimizado) / fitness_experto) * 100
    mejora_experto_vs_default = ((fitness_default - fitness_experto) / fitness_default) * 100
    
    # Mostrar resultados
    print("\nResultados de la comparación:")
    print(f"Fitness configuración por defecto: {fitness_default:.2f}")
    print(f"Fitness configuración humana experta: {fitness_experto:.2f}")
    print(f"Fitness configuración AG: {fitness_optimizado:.2f}")
    print(f"Mejora AG vs Default: {mejora_ag_vs_default:.2f}%")
    print(f"Mejora AG vs Experto: {mejora_ag_vs_experto:.2f}%")
    print(f"Mejora Experto vs Default: {mejora_experto_vs_default:.2f}%")
    
    # Visualizar las tres configuraciones
    visualizar_tres_configuraciones(config_default, config_experta, config_ag, contador_productos)
    
    return {
        "fitness_default": fitness_default,
        "fitness_experto": fitness_experto,
        "fitness_optimizado": fitness_optimizado,
        "config_experta": config_experta
    }

def visualizar_tres_configuraciones(config_default, config_experta, config_ag, frecuencias):
    """
    Visualiza las tres configuraciones en un gráfico de comparación
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Configuraciones
    configs = [config_default, config_experta, config_ag]
    titulos = ["Configuración por defecto", "Configuración humana experta", "Configuración AG"]
    
    # Para cada configuración
    for i, (config, titulo) in enumerate(zip(configs, titulos)):
        almacen = Almacen(configuracion=config)
        
        mapa_calor = np.zeros((almacen.filas, almacen.columnas))
        
        # Valor máximo para normalizar el mapa de calor
        max_frecuencia = max(frecuencias.values()) if frecuencias else 1
        
        # Llenar el mapa de calor
        for estanteria, producto in almacen.estanterias.items():
            fila, col = almacen.posiciones_estanterias[estanteria]
            frecuencia = frecuencias.get(producto, 0)
            mapa_calor[fila, col] = frecuencia / max_frecuencia
        
        # Visualizar
        ax = axes[i]
        cmap = plt.cm.get_cmap('coolwarm')
        im = ax.imshow(mapa_calor, cmap=cmap, interpolation='nearest')
        
        # Añadir texto
        for estanteria, producto in almacen.estanterias.items():
            fila, col = almacen.posiciones_estanterias[estanteria]
            frecuencia = frecuencias.get(producto, 0)
            ax.text(col, fila, f"{producto}\n({frecuencia})", 
                    ha='center', va='center', 
                    color='black' if mapa_calor[fila, col] < 0.6 else 'white', 
                    fontsize=7)
        
        # Marcar zona de carga
        ax.text(almacen.carga_pos[1], almacen.carga_pos[0], 'C', 
                ha='center', va='center', color='white', 
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle="round", facecolor='blue', alpha=0.7))
        
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        ax.set_title(titulo)
    
    # Añadir una barra de color común
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Frecuencia normalizada')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

def calcular_pasos_configuracion(configuracion, historico_ordenes, max_ordenes=None, max_iteraciones_recocido=100):
    """
    Calcula la cantidad total de pasos para procesar un conjunto de órdenes con una configuración de almacén.
    
    Args:
        configuracion: Lista con la configuración de productos en las estanterías
        historico_ordenes: Lista de órdenes históricas
        max_ordenes: Número máximo de órdenes a evaluar (None para todas)
        max_iteraciones_recocido: Número máximo de iteraciones para el recocido simulado
    
    Returns:
        dict: Diccionario con los resultados, incluyendo pasos totales, promedio, y detalle por orden
    """
    
    # Limitar el número de órdenes a evaluar si es necesario
    ordenes_a_evaluar = historico_ordenes
    if max_ordenes is not None and max_ordenes < len(historico_ordenes):
        ordenes_a_evaluar = historico_ordenes[:max_ordenes]
    
    almacen = Almacen(configuracion=configuracion)
    
    pasos_por_orden = []
    pasos_totales = 0
    
    for i, orden in enumerate(ordenes_a_evaluar):
        optimizador = OptimizadorRecocido(almacen, orden)
        _, pasos = optimizador.ejecutar(max_iteraciones=max_iteraciones_recocido)
        pasos_por_orden.append(pasos)
        pasos_totales += pasos
        
        if (i + 1) % 10 == 0:
            print(f"  Procesadas {i + 1}/{len(ordenes_a_evaluar)} órdenes...")
    
    pasos_promedio = pasos_totales / len(ordenes_a_evaluar)
    
    return {
        "pasos_totales": pasos_totales,
        "pasos_promedio": pasos_promedio,
        "pasos_por_orden": pasos_por_orden,
        "num_ordenes": len(ordenes_a_evaluar)
    }

def comparar_pasos_configuraciones(historico_ordenes, config_original, config_nueva, max_ordenes=50):
    """
    Compara la cantidad de pasos entre dos configuraciones de almacén.
    
    Args:
        historico_ordenes: Lista de órdenes históricas
        config_original: Configuración original del almacén
        config_nueva: Nueva configuración del almacén a comparar
        max_ordenes: Número máximo de órdenes a evaluar
    
    Returns:
        dict: Diccionario con los resultados de la comparación
    """
    print("Calculando pasos para la configuración original...")
    resultados_original = calcular_pasos_configuracion(
        config_original, 
        historico_ordenes, 
        max_ordenes=max_ordenes
    )
    
    print("Calculando pasos para la nueva configuración...")
    resultados_nueva = calcular_pasos_configuracion(
        config_nueva, 
        historico_ordenes, 
        max_ordenes=max_ordenes
    )
    
    # Calcular diferencias
    diferencia_total = resultados_original["pasos_totales"] - resultados_nueva["pasos_totales"]
    diferencia_promedio = resultados_original["pasos_promedio"] - resultados_nueva["pasos_promedio"]
    
    # Calcular porcentajes de mejora
    if resultados_original["pasos_totales"] > 0:
        porcentaje_mejora = (diferencia_total / resultados_original["pasos_totales"]) * 100
    else:
        porcentaje_mejora = 0
    
    # Calcular mejoras a nivel de orden individual
    mejoras_por_orden = []
    for i in range(len(resultados_original["pasos_por_orden"])):
        pasos_original = resultados_original["pasos_por_orden"][i]
        pasos_nueva = resultados_nueva["pasos_por_orden"][i]
        diferencia = pasos_original - pasos_nueva
        
        if pasos_original > 0:
            porcentaje = (diferencia / pasos_original) * 100
        else:
            porcentaje = 0
            
        mejoras_por_orden.append({
            "orden_num": i,
            "pasos_original": pasos_original,
            "pasos_nueva": pasos_nueva,
            "diferencia": diferencia,
            "porcentaje_mejora": porcentaje
        })
    
    # Ordenar las mejoras de mayor a menor
    mejoras_ordenadas = sorted(mejoras_por_orden, key=lambda x: x["porcentaje_mejora"], reverse=True)
    
    return {
        "resultados_original": resultados_original,
        "resultados_nueva": resultados_nueva,
        "diferencia_total": diferencia_total,
        "diferencia_promedio": diferencia_promedio,
        "porcentaje_mejora": porcentaje_mejora,
        "mejoras_por_orden": mejoras_por_orden,
        "mejoras_ordenadas": mejoras_ordenadas
    }

def visualizar_comparacion_pasos(resultados_comparacion):
    """
    Visualiza la comparación de pasos entre dos configuraciones.
    
    Args:
        resultados_comparacion: Resultados de la función comparar_pasos_configuraciones
    """
    # Imprimir resumen de la comparación
    print("\n===== RESUMEN DE LA COMPARACIÓN DE PASOS =====")
    print(f"Número de órdenes evaluadas: {resultados_comparacion['resultados_original']['num_ordenes']}")
    print(f"Pasos totales (configuración original): {resultados_comparacion['resultados_original']['pasos_totales']:.2f}")
    print(f"Pasos totales (nueva configuración): {resultados_comparacion['resultados_nueva']['pasos_totales']:.2f}")
    print(f"Diferencia total de pasos: {resultados_comparacion['diferencia_total']:.2f}")
    print(f"Pasos promedio (configuración original): {resultados_comparacion['resultados_original']['pasos_promedio']:.2f}")
    print(f"Pasos promedio (nueva configuración): {resultados_comparacion['resultados_nueva']['pasos_promedio']:.2f}")
    print(f"Diferencia promedio de pasos: {resultados_comparacion['diferencia_promedio']:.2f}")
    print(f"Porcentaje de mejora: {resultados_comparacion['porcentaje_mejora']:.2f}%")
    
    # Calcular estadísticas adicionales
    mejoras = [m["porcentaje_mejora"] for m in resultados_comparacion["mejoras_por_orden"]]
    
    print("\n===== ESTADÍSTICAS DE MEJORA POR ORDEN =====")
    print(f"Mejora máxima: {max(mejoras):.2f}%")
    print(f"Mejora mínima: {min(mejoras):.2f}%")
    print(f"Desviación estándar: {np.std(mejoras):.2f}%")
    
    # Contar órdenes mejoradas, empeoradas y sin cambios
    mejoradas = sum(1 for m in mejoras if m > 0)
    empeoradas = sum(1 for m in mejoras if m < 0)
    sin_cambios = sum(1 for m in mejoras if m == 0)
    
    print(f"Órdenes mejoradas: {mejoradas} ({mejoradas/len(mejoras)*100:.2f}%)")
    print(f"Órdenes empeoradas: {empeoradas} ({empeoradas/len(mejoras)*100:.2f}%)")
    print(f"Órdenes sin cambios: {sin_cambios} ({sin_cambios/len(mejoras)*100:.2f}%)")
    
    # Mostrar las 5 mejoras más significativas
    print("\n===== TOP 5 MEJORAS MÁS SIGNIFICATIVAS =====")
    for i, mejora in enumerate(resultados_comparacion["mejoras_ordenadas"][:5]):
        print(f"{i+1}. Orden #{mejora['orden_num']}: Reducción de {mejora['pasos_original']:.2f} a {mejora['pasos_nueva']:.2f} pasos ({mejora['porcentaje_mejora']:.2f}% de mejora)")
    
    # Mostrar las 5 peores
    if empeoradas > 0:
        print("\n===== TOP 5 CASOS DE EMPEORAMIENTO =====")
        peores = sorted(resultados_comparacion["mejoras_por_orden"], key=lambda x: x["porcentaje_mejora"])
        for i, empeora in enumerate(peores[:5]):
            print(f"{i+1}. Orden #{empeora['orden_num']}: Aumento de {empeora['pasos_original']:.2f} a {empeora['pasos_nueva']:.2f} pasos ({abs(empeora['porcentaje_mejora']):.2f}% de empeoramiento)")
    
    # Crear visualizaciones
    plt.figure(figsize=(15, 10))
    
    # Histograma de porcentajes de mejora
    plt.subplot(2, 2, 1)
    plt.hist(mejoras, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribución de los porcentajes de mejora por orden')
    plt.xlabel('Porcentaje de mejora')
    plt.ylabel('Frecuencia')
    
    # Gráfico de dispersión de pasos original vs nuevo
    plt.subplot(2, 2, 2)
    original = [m["pasos_original"] for m in resultados_comparacion["mejoras_por_orden"]]
    nueva = [m["pasos_nueva"] for m in resultados_comparacion["mejoras_por_orden"]]
    plt.scatter(original, nueva, alpha=0.5)
    max_val = max(max(original), max(nueva))
    plt.plot([0, max_val], [0, max_val], 'r--')  # Línea diagonal
    plt.title('Pasos originales vs. Pasos nuevos')
    plt.xlabel('Pasos en configuración original')
    plt.ylabel('Pasos en nueva configuración')
    plt.grid(True)
    
    # Gráfico de barras comparativas para las 10 primeras órdenes
    plt.subplot(2, 1, 2)
    indices = range(min(10, len(mejoras)))
    bar_width = 0.35
    plt.bar([i for i in indices], 
            [resultados_comparacion["mejoras_por_orden"][i]["pasos_original"] for i in indices],
            bar_width, label='Configuración original', color='indianred')
    plt.bar([i + bar_width for i in indices], 
            [resultados_comparacion["mejoras_por_orden"][i]["pasos_nueva"] for i in indices],
            bar_width, label='Nueva configuración', color='forestgreen')
    plt.xlabel('Número de orden')
    plt.ylabel('Número de pasos')
    plt.title('Comparación de pasos para las primeras 10 órdenes')
    plt.xticks([i + bar_width/2 for i in indices], [i for i in indices])
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Gráfico adicional: Mejora acumulada
    plt.figure(figsize=(10, 6))
    diferencias_acumuladas = np.cumsum([m["diferencia"] for m in resultados_comparacion["mejoras_por_orden"]])
    plt.plot(diferencias_acumuladas, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Diferencia acumulada de pasos')
    plt.xlabel('Número de orden')
    plt.ylabel('Pasos ahorrados (acumulado)')
    plt.grid(True)
    plt.show()

# Ejemplo de uso de las funciones
def evaluar_todas_configuraciones(historico_ordenes, max_ordenes=30):
    """
    Evalúa y compara todas las configuraciones: por defecto, humana experta y algoritmo genético.
    
    Args:
        historico_ordenes: Lista de órdenes históricas
        max_ordenes: Número máximo de órdenes a evaluar
    """
    print("Iniciando evaluación completa de configuraciones...")
    
    # Configuración por defecto (productos secuenciales)
    config_default = list(range(1, 49))
    
    # Calcular frecuencias de productos
    contador_productos = Counter()
    for orden in historico_ordenes:
        contador_productos.update(orden)
    
    # Crear configuración humana experta
    config_experta = crear_configuracion_experta(contador_productos)
    
    # Ejecutar algoritmo genético
    print("Ejecutando algoritmo genético...")
    ag = AlgoritmoGenetico(
        historico_ordenes=historico_ordenes,
        tam_poblacion=30,
        prob_cruce=0.9,
        prob_mutacion=0.3,
        num_elitismo=2,
        max_generaciones=30  # Reducido para pruebas
    )
    
    resultado_ag = ag.ejecutar()
    config_ag = resultado_ag['mejor_individuo']
    
    # Comparar configuraciones por pasos
    print("\n1. Comparando configuración por defecto vs. humana experta:")
    resultados_default_vs_experta = comparar_pasos_configuraciones(
        historico_ordenes, 
        config_default, 
        config_experta, 
        max_ordenes=max_ordenes
    )
    visualizar_comparacion_pasos(resultados_default_vs_experta)
    
    print("\n2. Comparando configuración por defecto vs. algoritmo genético:")
    resultados_default_vs_ag = comparar_pasos_configuraciones(
        historico_ordenes, 
        config_default, 
        config_ag, 
        max_ordenes=max_ordenes
    )
    visualizar_comparacion_pasos(resultados_default_vs_ag)
    
    print("\n3. Comparando configuración humana experta vs. algoritmo genético:")
    resultados_experta_vs_ag = comparar_pasos_configuraciones(
        historico_ordenes, 
        config_experta, 
        config_ag, 
        max_ordenes=max_ordenes
    )
    visualizar_comparacion_pasos(resultados_experta_vs_ag)
    
    # Visualizar las tres configuraciones
    visualizar_tres_configuraciones(config_default, config_experta, config_ag, contador_productos)
    
    # Mostrar resumen final
    print("\n===== RESUMEN FINAL DE COMPARACIÓN =====")
    print(f"Mejora configuración experta vs. por defecto: {resultados_default_vs_experta['porcentaje_mejora']:.2f}%")
    print(f"Mejora algoritmo genético vs. por defecto: {resultados_default_vs_ag['porcentaje_mejora']:.2f}%")
    print(f"Mejora algoritmo genético vs. experta: {resultados_experta_vs_ag['porcentaje_mejora']:.2f}%")
    
    return {
        "config_default": config_default,
        "config_experta": config_experta,
        "config_ag": config_ag,
        "resultados_default_vs_experta": resultados_default_vs_experta,
        "resultados_default_vs_ag": resultados_default_vs_ag,
        "resultados_experta_vs_ag": resultados_experta_vs_ag
    }

if __name__ == "__main__":
    archivo_ordenes = "ordenes.csv"
    print(f"Cargando órdenes históricas desde {archivo_ordenes}...")
    
    try:
        historico_ordenes = cargar_ordenes_historicas(archivo_ordenes)
        print(f"Se cargaron {len(historico_ordenes)} órdenes históricas.")
    except:
        print("No se pudo cargar el archivo. Usando órdenes de ejemplo...")
        historico_ordenes = [
            [5, 10, 15, 20],
            [1, 3, 5, 7, 9],
            [2, 4, 6, 8, 10],
            [11, 22, 33, 44],
            [12, 24, 36, 48]
        ]
    
    resultados = evaluar_todas_configuraciones(historico_ordenes, max_ordenes=20)

    print("\nAnalizando órdenes históricas...")
    todos_productos = set()
    for orden in historico_ordenes:
        todos_productos.update(orden)
    
    contador_productos = Counter()
    for orden in historico_ordenes:
        contador_productos.update(orden)
    
    print(f"Total de productos únicos en las órdenes: {len(todos_productos)}")
    print(f"Número máximo de productos en una orden: {max(len(o) for o in historico_ordenes)}")
    print(f"Número mínimo de productos en una orden: {min(len(o) for o in historico_ordenes)}")
    print(f"Promedio de productos por orden: {sum(len(o) for o in historico_ordenes) / len(historico_ordenes):.2f}")
    
    print("\nProductos más frecuentes:")
    for producto, frecuencia in contador_productos.most_common(5):
        print(f"Producto {producto}: {frecuencia} veces")
    
    print("\nIniciando algoritmo genético...")
    ag = AlgoritmoGenetico(
        historico_ordenes=historico_ordenes,
        tam_poblacion=40,
        prob_cruce=0.92,
        prob_mutacion=0.35,
        num_elitismo=1,
        max_generaciones=5
    )
    
    resultado = ag.ejecutar()
    
    print("\nVisualizando evolución del fitness...")
    ag.visualizar_evolucion()
    
    print("\nVisualizando configuración optimizada del almacén...")
    ag.visualizar_almacen_optimizado()
    
    print("\nComparando con la configuración por defecto...")
    config_default = list(range(1, 49))
    
    almacen_default = Almacen(configuracion=config_default)
    costo_total_default = 0
    
    for orden in historico_ordenes[:50]:
        optimizador = OptimizadorRecocido(almacen_default, orden)
        _, costo = optimizador.ejecutar(max_iteraciones=100)
        costo_total_default += costo
    
    fitness_default = costo_total_default / 50
    
    almacen_optimizado = Almacen(configuracion=resultado['mejor_individuo'])
    costo_total_optimizado = 0
    
    for orden in historico_ordenes[:50]:
        optimizador = OptimizadorRecocido(almacen_optimizado, orden)
        _, costo = optimizador.ejecutar(max_iteraciones=100)
        costo_total_optimizado += costo
    
    fitness_optimizado = costo_total_optimizado / 50
    
    mejora_porcentaje = ((fitness_default - fitness_optimizado) / fitness_default) * 100
    print(f"Fitness configuración por defecto: {fitness_default:.2f}")
    print(f"Fitness configuración optimizada: {fitness_optimizado:.2f}")
    print(f"Mejora: {mejora_porcentaje:.2f}%")
    
    # Mostrar resultados finales
    print("\nResultados finales:")
    print(f"Mejor configuración: {resultado['mejor_individuo']}")
    print(f"Fitness (costo promedio): {resultado['mejor_fitness']:.2f}")
    print(f"Tiempo total de ejecución: {resultado['tiempo_total']:.2f} segundos")

    print("\nRealizando comparación de configuraciones...")
    resultado_comparacion = comparar_configuraciones(historico_ordenes, resultado)
    
    # Imprimir la configuración experta para referencia
    print("\nConfiguración humana experta:")
    print(resultado_comparacion["config_experta"])