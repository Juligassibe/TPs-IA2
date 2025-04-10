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

        # Filtrar solo las posiciones válidas (pasillos)
        return [pos for pos in adyacentes if self.es_valido(pos)]


class Montacargas:
    def __init__(self, id: int, posicion: Tuple[int, int], prioridad: int = 0):
        """
        Inicializa un montacargas con un ID, posición inicial y prioridad.
        """
        self.id = id
        self.posicion = posicion
        self.prioridad = prioridad
        self.camino = []
        self.camino_historico = [posicion]
        self.destino_producto = None
        self.paso_actual = 0
        self.en_movimiento = False
        self.esperando = False

    def asignar_camino(self, camino: List[Tuple[int, int]], destino_producto: int):
        """Asigna un nuevo camino al montacargas"""
        self.camino = camino
        self.destino_producto = destino_producto
        self.paso_actual = 0
        self.en_movimiento = True
        self.esperando = False

    def obtener_posicion_actual(self) -> Tuple[int, int]:
        """Devuelve la posición actual del montacargas"""
        return self.posicion

    def obtener_siguiente_posicion(self) -> Optional[Tuple[int, int]]:
        """Devuelve la siguiente posición a la que se moverá el montacargas"""
        if not self.en_movimiento or self.esperando:
            return None

        if self.paso_actual + 1 < len(self.camino):
            return self.camino[self.paso_actual + 1]
        return None

    def mover_siguiente(self) -> bool:
        """
        Avanza el montacargas a la siguiente posición en su camino.
        Retorna True si se puede seguir moviendo, False si ha llegado al destino.
        """
        if not self.en_movimiento or self.esperando:
            return False

        if self.paso_actual + 1 < len(self.camino):
            self.paso_actual += 1
            self.posicion = self.camino[self.paso_actual]
            # Registrar la nueva posición en el historial
            self.camino_historico.append(self.posicion)
            return True
        else:
            self.en_movimiento = False
            return False

    def ha_llegado_destino(self) -> bool:
        """Verifica si el montacargas ha llegado a su destino"""
        return self.paso_actual == len(self.camino) - 1 if self.camino else False


class SistemaMultiagente:
    def __init__(self, almacen: Almacen):
        self.almacen = almacen
        self.montacargas = {}
        self.agente_a_estrella = AgenteAEstrella(almacen)
        self.tiempo_actual = 0
        self.colisiones_evitadas = 0

    def agregar_montacargas(self, id: int, posicion: Optional[Tuple[int, int]] = None, prioridad: int = 0):
        """
        Agrega un nuevo montacargas al sistema.
        Si no se especifica posición, se coloca en la zona de carga.
        """
        if posicion is None:
            posicion = self.almacen.carga_pos

        self.montacargas[id] = Montacargas(id, posicion, prioridad)
        return self.montacargas[id]

    def asignar_ruta(self, id_montacargas: int, destino_producto: int):
        """
        Asigna una ruta a un montacargas para llegar a un producto.
        """
        if id_montacargas not in self.montacargas:
            print(f"Error: No existe el montacargas con ID {id_montacargas}")
            return

        montacargas = self.montacargas[id_montacargas]
        posicion_inicial = montacargas.obtener_posicion_actual()

        # Encuentra un camino desde la posición actual hasta el producto
        camino = self.agente_a_estrella.encontrar_camino(posicion_inicial, destino_producto)

        if camino:
            montacargas.asignar_camino(camino, destino_producto)
            print(f"Ruta asignada a montacargas {id_montacargas} hacia producto {destino_producto}")

    def detectar_colision(self) -> List[Tuple[int, int]]:
        """
        Detecta posibles colisiones entre montacargas.
        Retorna lista de pares de IDs de montacargas que colisionarían.
        """
        colisiones = []
        ids_montacargas = list(self.montacargas.keys())

        for i in range(len(ids_montacargas)):
            for j in range(i + 1, len(ids_montacargas)):
                m1 = self.montacargas[ids_montacargas[i]]
                m2 = self.montacargas[ids_montacargas[j]]

                # Si alguno de los montacargas no está en movimiento o está esperando, no habrá colisión
                if not m1.en_movimiento or not m2.en_movimiento or m1.esperando or m2.esperando:
                    continue

                # Colisión si se cruzan (uno va a donde el otro está y viceversa)
                if (m1.obtener_siguiente_posicion() == m2.obtener_posicion_actual() and
                        m2.obtener_siguiente_posicion() == m1.obtener_posicion_actual()):
                    colisiones.append((m1.id, m2.id))
                    continue

                # Colisión si ambos van a la misma posición
                if (m1.obtener_siguiente_posicion() is not None and
                        m2.obtener_siguiente_posicion() is not None and
                        m1.obtener_siguiente_posicion() == m2.obtener_siguiente_posicion()):
                    colisiones.append((m1.id, m2.id))

                # Colisión si uno ya está donde el otro quiere ir
                if (m1.obtener_posicion_actual() == m2.obtener_siguiente_posicion() or
                        m2.obtener_posicion_actual() == m1.obtener_siguiente_posicion()):
                    colisiones.append((m1.id, m2.id))

        return colisiones

    def encontrar_casilla_evasion(self, montacargas, posiciones_ocupadas=None):
        """
        Encuentra una casilla adyacente válida para que el montacargas evada una colisión.
        Retorna la posición de la casilla o None si no hay opciones disponibles.
        """
        if posiciones_ocupadas is None:
            posiciones_ocupadas = []

        pos_actual = montacargas.obtener_posicion_actual()

        # Posibles movimientos (arriba, abajo, izquierda, derecha)
        movimientos = [
            (pos_actual[0] - 1, pos_actual[1]),  # Arriba
            (pos_actual[0] + 1, pos_actual[1]),  # Abajo
            (pos_actual[0], pos_actual[1] - 1),  # Izquierda
            (pos_actual[0], pos_actual[1] + 1)  # Derecha
        ]

        # Filtra solo las posiciones válidas y no ocupadas
        opciones_validas = []
        for pos in movimientos:
            if self.almacen.es_valido(pos) and pos not in posiciones_ocupadas:
                opciones_validas.append(pos)

        # Si hay opciones, retorna la posición con mejor heurística hacia el destino
        if opciones_validas:
            # Obtiene el destino final del camino actual
            if montacargas.camino and len(montacargas.camino) > 0:
                destino = montacargas.camino[-1]
                # Ordena opciones por distancia (heurística) al destino
                opciones_validas.sort(key=lambda pos:
                self.agente_a_estrella.heuristica(pos, destino))
                return opciones_validas[0]  # Retorna la mejor opción

        return None

    def mover_a_casilla_evasion(self, montacargas, casilla_evasion, posiciones_ocupadas=None):
        """
        Mueve el montacargas a una casilla de evasión y recalcula su ruta desde ahí.
        """
        if posiciones_ocupadas is None:
            posiciones_ocupadas = []

        # Actualiza la posición actual del montacargas
        montacargas.posicion = casilla_evasion
        # Registrar la nueva posición en el historial
        montacargas.camino_historico.append(casilla_evasion)

        # Recalcula la ruta desde la nueva posición evitando posiciones ocupadas
        nueva_ruta = self.agente_a_estrella.encontrar_camino(
            casilla_evasion,
            montacargas.destino_producto,
            posiciones_ocupadas
        )

        if nueva_ruta:
            # Asigna la nueva ruta
            montacargas.camino = nueva_ruta
            montacargas.paso_actual = 0
            montacargas.en_movimiento = True
            montacargas.esperando = False
            return True

        return False

    def resolver_colisiones(self, colisiones: List[Tuple[int, int]]):
        """
        Resuelve las colisiones haciendo que el montacargas de menor prioridad
        se mueva a una casilla adyacente o espere si no hay opciones.
        """
        for id1, id2 in colisiones:
            m1 = self.montacargas[id1]
            m2 = self.montacargas[id2]

            # Determina cuál montacargas tiene menor prioridad
            montacargas_menor_prioridad = m1 if m1.prioridad > m2.prioridad else m2
            montacargas_mayor_prioridad = m2 if montacargas_menor_prioridad == m1 else m1

            print(f"⚠️ Colisión detectada entre montacargas {id1} y {id2}")
            print(f"Montacargas {montacargas_menor_prioridad.id} (menor prioridad) buscará evadir la colisión")

            # Obtener posiciones ocupadas para evitar nuevas colisiones
            posiciones_ocupadas = []
            for otro_id, otro_montacargas in self.montacargas.items():
                if otro_id != montacargas_menor_prioridad.id:
                    posiciones_ocupadas.append(otro_montacargas.obtener_posicion_actual())

                    # También agregar la siguiente posición del montacargas de mayor prioridad
                    if otro_id == montacargas_mayor_prioridad.id and otro_montacargas.obtener_siguiente_posicion():
                        posiciones_ocupadas.append(otro_montacargas.obtener_siguiente_posicion())

            # Busca una casilla adyacente para evadir que no esté en el camino del montacargas de mayor prioridad
            casilla_evasion = self.encontrar_casilla_evasion(montacargas_menor_prioridad, posiciones_ocupadas)

            if casilla_evasion:
                # Si hay una casilla disponible, mover el montacargas ahí
                print(
                    f"Montacargas {montacargas_menor_prioridad.id} se moverá a la casilla {casilla_evasion} para evadir")
                exito = self.mover_a_casilla_evasion(montacargas_menor_prioridad, casilla_evasion, posiciones_ocupadas)
                if exito:
                    print(f"✅ Montacargas {montacargas_menor_prioridad.id} ha evadido la colisión con éxito")
                else:
                    print(
                        f"⚠️ Montacargas {montacargas_menor_prioridad.id} no pudo recalcular ruta desde la casilla de evasión")
                    montacargas_menor_prioridad.esperando = True
            else:
                # Si no hay casillas disponibles, hacer que espere
                print(
                    f"⚠️ No hay casillas adyacentes disponibles para evasión. Montacargas {montacargas_menor_prioridad.id} esperará un turno")
                montacargas_menor_prioridad.esperando = True

            self.colisiones_evitadas += 1

    def simular_paso(self):
        """
        Simula un paso en el sistema, moviendo los montacargas y resolviendo colisiones.
        """
        self.tiempo_actual += 1

        # Detectar y resolver colisiones potenciales antes de mover los montacargas
        colisiones = self.detectar_colision()
        if colisiones:
            self.resolver_colisiones(colisiones)

        # Mover montacargas si no están esperando
        for id_montacargas, montacargas in self.montacargas.items():
            if montacargas.en_movimiento and not montacargas.esperando:
                sigue_moviendo = montacargas.mover_siguiente()
                if not sigue_moviendo:
                    print(
                        f"✅ Montacargas {id_montacargas} ha llegado a su destino: producto {montacargas.destino_producto}")
            elif montacargas.esperando:
                # Después de un paso de espera, puede continuar
                montacargas.esperando = False

    def simular_hasta_completar(self, max_pasos=100):
        """
        Simula el sistema hasta que todos los montacargas lleguen a su destino
        o hasta alcanzar el número máximo de pasos.
        """
        pasos = 0

        while pasos < max_pasos:
            # Verifica si todos los montacargas han llegado a su destino
            todos_llegaron = True
            for montacargas in self.montacargas.values():
                if montacargas.en_movimiento:
                    todos_llegaron = False
                    break

            if todos_llegaron:
                print(f"✅ Todos los montacargas han llegado a sus destinos en {pasos} pasos.")
                break

            self.simular_paso()
            pasos += 1

            print(f"\n--- Paso {pasos} ---")
            for id_montacargas, montacargas in self.montacargas.items():
                estado = "Esperando" if montacargas.esperando else "En movimiento" if montacargas.en_movimiento else "Detenido"
                print(f"Montacargas {id_montacargas}: {estado} en posición {montacargas.obtener_posicion_actual()}")

        if pasos >= max_pasos:
            print(f"⚠️ Se alcanzó el límite máximo de {max_pasos} pasos sin completar todas las rutas.")

        print(f"Se evitaron {self.colisiones_evitadas} colisiones durante la simulación.")

    def visualizar_estado(self):
        """
        Visualiza el estado actual del almacén con los montacargas y sus rutas históricas,
        mostrando las coordenadas de fila y columna.
        """
        visualizacion = np.zeros((self.almacen.filas, self.almacen.columnas))

        # Códigos para visualización:
        # 0: Pasillo vacío
        # 1: Estantería
        # 2: Camino histórico montacargas 1
        # 3: Camino histórico montacargas 2
        # 4: Posición actual montacargas 1
        # 5: Posición actual montacargas 2
        # 6: Posición destino montacargas 1
        # 7: Posición destino montacargas 2
        # 8: Zona de carga
        # 9: Camino actual planificado montacargas 1 (opcional)
        # 10: Camino actual planificado montacargas 2 (opcional)

        # Marca estanterías
        for producto, pos in self.almacen.estanterias.items():
            visualizacion[pos] = 1

        # Marca zona de carga
        visualizacion[self.almacen.carga_pos] = 8

        # Marca caminos y posiciones de montacargas
        for i, (id_montacargas, montacargas) in enumerate(self.montacargas.items()):
            if i >= 2:  # Solo visualizamos 2 montacargas por simplicidad
                break

            # Marcar el camino histórico primero (para que pueda ser sobrescrito por posiciones actuales)
            for pos in montacargas.camino_historico:
                # No sobrescribir estanterías o zona de carga
                if visualizacion[pos] == 0 or (visualizacion[pos] >= 2 and visualizacion[pos] <= 3):
                    visualizacion[pos] = 2 + i  # 2 para montacargas 1, 3 para montacargas 2

            # Marca posición actual
            pos_actual = montacargas.obtener_posicion_actual()
            visualizacion[pos_actual] = 4 + i  # 4 para montacargas 1, 5 para montacargas 2

            # Marca destino (última posición del camino o adyacente al producto)
            if montacargas.destino_producto is not None:
                pos_destino = None
                if montacargas.camino and len(montacargas.camino) > 0:
                    pos_destino = montacargas.camino[-1]

                if pos_destino:
                    visualizacion[pos_destino] = 6 + i  # 6 para destino mont. 1, 7 para destino mont. 2

        # Gráfica
        plt.figure(figsize=(12, 10))
        cmap = plt.cm.colors.ListedColormap(['white', 'lightgray', 'lightgreen', 'lightblue',
                                             'green', 'blue', 'darkgreen', 'darkblue', 'yellow'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(visualizacion, cmap=cmap, norm=norm)

        # Dibuja la cuadrícula
        plt.grid(color='black', linestyle='-', linewidth=0.5)

        # Agrega coordenadas numéricas en cada celda (fila,columna)
        for f in range(self.almacen.filas):
            for c in range(self.almacen.columnas):
                # Si es una estantería o la zona de carga, no mostrar coordenadas
                if visualizacion[f, c] != 1 and visualizacion[f, c] != 8:
                    plt.text(c, f, f"({f},{c})", ha='center', va='center',
                             color='black', fontsize=7, alpha=0.5)

        # Agrega etiquetas de productos
        for producto, (f, c) in self.almacen.estanterias.items():
            plt.text(c, f, str(producto), ha='center', va='center', color='black', fontsize=9, fontweight='bold')

        # Etiqueta zona de carga
        plt.text(self.almacen.carga_pos[1], self.almacen.carga_pos[0], 'C',
                 ha='center', va='center', color='black', fontweight='bold')

        # Etiqueta posiciones de montacargas
        for i, (id_montacargas, montacargas) in enumerate(self.montacargas.items()):
            if i >= 2:  # Solo visualizamos 2 montacargas por simplicidad
                break

            pos = montacargas.obtener_posicion_actual()
            plt.text(pos[1], pos[0], f"M{montacargas.id}", ha='center', va='center',
                     color='white', fontweight='bold', fontsize=10)

        # Configuración de ejes
        plt.xticks(np.arange(-0.5, self.almacen.columnas, 1), [])
        plt.yticks(np.arange(-0.5, self.almacen.filas, 1), [])
        plt.xlim(-0.5, self.almacen.columnas - 0.5)
        plt.ylim(-0.5, self.almacen.filas - 0.5)

        # Título y leyenda
        plt.title(f"Estado de los montacargas en el almacén - Paso {self.tiempo_actual}", fontsize=12)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Pasillo'),
            Patch(facecolor='lightgray', edgecolor='black', label='Estantería'),
            Patch(facecolor='lightgreen', edgecolor='black', label='Camino montacargas 1'),
            Patch(facecolor='lightblue', edgecolor='black', label='Camino montacargas 2'),
            Patch(facecolor='green', edgecolor='black', label='Posición montacargas 1'),
            Patch(facecolor='blue', edgecolor='black', label='Posición montacargas 2'),
            Patch(facecolor='darkgreen', edgecolor='black', label='Destino montacargas 1'),
            Patch(facecolor='darkblue', edgecolor='black', label='Destino montacargas 2'),
            Patch(facecolor='yellow', edgecolor='black', label='Zona de carga')
        ]
        plt.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.05), ncol=3)

        plt.tight_layout()
        plt.show()


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

    def encontrar_camino(self, inicio: Optional[Tuple[int, int]], destino_producto: int,
                         posiciones_ocupadas: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Encuentra el camino más corto desde 'inicio' hasta alguna de las posiciones
        horizontalmente adyacentes al 'destino_producto' usando A*.

        Se añade el parámetro 'posiciones_ocupadas' para evitar colisiones.
        """
        # Si no se proporciona posición inicial, usa la zona de carga
        if inicio is None:
            inicio = self.almacen.carga_pos

        # Si no se proporciona lista de posiciones ocupadas, usar lista vacía
        if posiciones_ocupadas is None:
            posiciones_ocupadas = []

        # Obtiene posiciones adyacentes válidas al producto destino
        destinos_posibles = self.almacen.get_posiciones_adyacentes(destino_producto)

        if not destinos_posibles:
            print(f"No hay posiciones adyacentes accesibles para el producto {destino_producto}.")
            return []

        # Lista abierta para el algoritmo A* (cola de prioridad)
        # Cada elemento de la cola: (f, contador, posicion, padre, g)
        abierta = []
        contador = 0

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
                # Verifica si el movimiento es válido y no está ocupado por otro montacargas
                if not self.almacen.es_valido(siguiente_pos) or siguiente_pos in posiciones_ocupadas:
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

        # Dibujar la cuadrícula
        plt.grid(color='black', linestyle='-', linewidth=0.5)

        # Agregar coordenadas numéricas en cada celda (fila,columna)
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

        # Agregar número de paso en cada celda del camino
        for i, pos in enumerate(camino):
            if i > 0 and i < len(camino) - 1:  # No mostrar en inicio ni fin
                plt.text(pos[1], pos[0], str(i), ha='center', va='center',
                         color='white', fontsize=9, fontweight='bold')

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
    sistema = SistemaMultiagente(almacen)

    # Menú principal
    while True:
        print("\n=== SISTEMA DE MONTACARGAS MÚLTIPLES ===")
        print("1. Agregar montacargas")
        print("2. Asignar tarea a montacargas")
        print("3. Simular sistema")
        print("4. Visualizar estado actual")
        print("5. Salir")

        opcion = input("\nSeleccione una opción: ")

        if opcion == "1":
            # Agrega montacargas
            try:
                id_montacargas = int(input("Ingrese ID para el nuevo montacargas: "))
                if id_montacargas in sistema.montacargas:
                    print(f"Error: Ya existe un montacargas con ID {id_montacargas}")
                    continue

                posicion_input = input(
                    "Ingrese posición inicial (fila,columna) o presione Enter para la zona de carga: ")
                if posicion_input.strip():
                    try:
                        fila_str, col_str = posicion_input.split(',')
                        fila = int(fila_str)
                        col = int(col_str)
                        posicion = (fila, col)
                        if not almacen.es_valido(posicion):
                            print("Posición inválida. Se usará la zona de carga.")
                            posicion = None
                    except:
                        print("Formato inválido. Se usará la zona de carga.")
                        posicion = None
                else:
                    posicion = None

                prioridad = int(input("Ingrese prioridad (menor número = mayor prioridad): "))

                montacargas = sistema.agregar_montacargas(id_montacargas, posicion, prioridad)
                print(
                    f"✅ Montacargas {id_montacargas} agregado en posición {montacargas.posicion} con prioridad {prioridad}")

            except ValueError:
                print("Error: Asegúrese de ingresar valores numéricos válidos")

        elif opcion == "2":
            # Asigna tarea a montacargas
            if not sistema.montacargas:
                print("Error: No hay montacargas en el sistema. Agregue al menos uno.")
                continue

            try:
                print("\nMontacargas disponibles:")
                for id_montacargas, montacargas in sistema.montacargas.items():
                    estado = "En espera" if not montacargas.en_movimiento else "En movimiento"
                    print(f"ID: {id_montacargas}, Posición: {montacargas.posicion}, Estado: {estado}")

                id_montacargas = int(input("\nSeleccione ID del montacargas: "))
                if id_montacargas not in sistema.montacargas:
                    print(f"Error: No existe un montacargas con ID {id_montacargas}")
                    continue

                producto_destino = int(input("Ingrese el número de producto a buscar (1-48): "))
                if not (1 <= producto_destino <= 48):
                    print("Error: El número de producto debe estar entre 1 y 48.")
                    continue

                sistema.asignar_ruta(id_montacargas, producto_destino)

            except ValueError:
                print("Error: Asegúrese de ingresar valores numéricos válidos")

        elif opcion == "3":
            # Simula sistema
            if not sistema.montacargas:
                print("Error: No hay montacargas en el sistema. Agregue al menos uno.")
                continue

            # Verificar si hay montacargas con tareas asignadas
            hay_montacargas_activos = False
            for montacargas in sistema.montacargas.values():
                if montacargas.en_movimiento:
                    hay_montacargas_activos = True
                    break

            if not hay_montacargas_activos:
                print("Error: No hay montacargas con tareas asignadas. Asigne al menos una tarea.")
                continue

            try:
                max_pasos = int(input("Ingrese máximo de pasos para la simulación: "))
                sistema.simular_hasta_completar(max_pasos)
                sistema.visualizar_estado()
            except ValueError:
                print("Error: Asegúrese de ingresar un valor numérico válido para los pasos")

        elif opcion == "4":
            # Visualiza estado actual
            if not sistema.montacargas:
                print("Error: No hay montacargas en el sistema. Agregue al menos uno.")
                continue

            sistema.visualizar_estado()

        elif opcion == "5":
            print("Saliendo del programa...")
            break

        else:
            print("Opción inválida. Intente nuevamente.")


if __name__ == "__main__":
    main()