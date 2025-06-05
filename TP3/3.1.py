import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #Para dividir datos, normalizar y calcular métricas
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns #Para mejorar el estilo de gráficos
import random
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos, Configura el estilo visual de los gráficos para que se vean mejor.
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RedNeuronalMulticapa:
    """
    Red Neuronal Multicapa para Regresión
    """
    
    def __init__(self, arquitectura, tasa_aprendizaje=0.01):
        """Inicializar la red neuronal"""
        self.arquitectura = arquitectura #Inicializa la red neuronal con una arquitectura específica. Por ejemplo, [1, 10, 5, 1] significa: 1 neurona de entrada, una capa oculta con 10 neuronas, otra con 5, y 1 neurona de salida.
        self.tasa_aprendizaje = tasa_aprendizaje
        self.num_capas = len(arquitectura) - 1
        
        # Inicializar pesos y sesgos
        self.pesos = {}
        self.sesgos = {}
        
        # Inicialización Xavier/Glorot para mejor convergencia
        for i in range(self.num_capas): #Crea los pesos (conexiones) entre neuronas usando la inicialización Xavier, que ayuda a que la red aprenda mejor. Los sesgos se inicializan en cero.
            self.pesos[f'W{i+1}'] = np.random.randn(arquitectura[i], arquitectura[i+1]) * np.sqrt(2.0 / arquitectura[i])
            self.sesgos[f'b{i+1}'] = np.zeros((1, arquitectura[i+1]))
        
        # Historial de entrenamiento
        self.historial_perdida = []
        
    def relu(self, z):
        """Función de activación ReLU"""
        return np.maximum(0, z)
    
    def relu_derivada(self, z):
        """Derivada de ReLU"""
        return (z > 0).astype(float)
    
    def propagacion_adelante(self, X):
        """Propagación hacia adelante"""
        activaciones = {'A0': X}
        
        for i in range(self.num_capas):
            z = np.dot(activaciones[f'A{i}'], self.pesos[f'W{i+1}']) + self.sesgos[f'b{i+1}']
            
            if i < self.num_capas - 1:  # Capas ocultas: ReLU
                activaciones[f'A{i+1}'] = self.relu(z)
            else:  # Capa de salida: Lineal
                activaciones[f'A{i+1}'] = z
            
            activaciones[f'Z{i+1}'] = z
        
        return activaciones
    
    def propagacion_atras(self, X, y, activaciones):
        """Retropropagación"""
        m = X.shape[0]
        gradientes = {}
        
        dA = activaciones[f'A{self.num_capas}'] - y
        
        for i in reversed(range(self.num_capas)):
            if i == self.num_capas - 1:
                dZ = dA
            else:
                dZ = dA * self.relu_derivada(activaciones[f'Z{i+1}'])
            
            gradientes[f'dW{i+1}'] = (1/m) * np.dot(activaciones[f'A{i}'].T, dZ)
            gradientes[f'db{i+1}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            if i > 0:
                dA = np.dot(dZ, self.pesos[f'W{i+1}'].T)
        
        return gradientes
    
    def actualizar_parametros(self, gradientes):
        """Actualizar pesos y sesgos"""
        for i in range(self.num_capas):
            self.pesos[f'W{i+1}'] -= self.tasa_aprendizaje * gradientes[f'dW{i+1}']
            self.sesgos[f'b{i+1}'] -= self.tasa_aprendizaje * gradientes[f'db{i+1}']
    
    def calcular_perdida(self, y_real, y_pred):
        """Calcular MSE"""
        return np.mean((y_real - y_pred) ** 2)
    
    def entrenar(self, X, y, epocas=1000, mostrar_progreso=False):
        """Entrenar la red neuronal"""
        if mostrar_progreso:
            print(f"Entrenando arquitectura {self.arquitectura}...")
        
        for epoca in range(epocas):
            activaciones = self.propagacion_adelante(X)
            y_pred = activaciones[f'A{self.num_capas}']
            
            perdida = self.calcular_perdida(y, y_pred)
            self.historial_perdida.append(perdida)
            
            gradientes = self.propagacion_atras(X, y, activaciones)
            self.actualizar_parametros(gradientes)
            
            if mostrar_progreso and (epoca % 200 == 0 or epoca == epocas - 1):
                print(f"Época {epoca:4d} | Pérdida: {perdida:.6f}")
    
    def predecir(self, X):
        """Hacer predicciones"""
        activaciones = self.propagacion_adelante(X)
        return activaciones[f'A{self.num_capas}']
    
    def evaluar(self, X, y):
        """Evaluar el modelo"""
        y_pred = self.predecir(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mse)
        
        return {'MSE': mse, 'RMSE': rmse, 'R²': r2}


class AlgoritmoGenetico:
    """
    Algoritmo Genético para optimizar hiperparámetros de Red Neuronal
    
    Cromosoma: [num_capas_ocultas, neuronas_capa1, neuronas_capa2, ..., tasa_aprendizaje_index]
    """
    
    def __init__(self, 
                 poblacion_size=20,
                 generaciones=10,
                 prob_mutacion=0.1,
                 prob_cruce=0.8,
                 max_capas=4,
                 max_neuronas=32,
                 epocas_entrenamiento=300):
        
        self.poblacion_size = poblacion_size
        self.generaciones = generaciones
        self.prob_mutacion = prob_mutacion
        self.prob_cruce = prob_cruce
        self.max_capas = max_capas
        self.max_neuronas = max_neuronas
        self.epocas_entrenamiento = epocas_entrenamiento
        
        # Tasas de aprendizaje predefinidas
        self.tasas_aprendizaje = [0.001, 0.01, 0.05, 0.1]
        
        self.mejor_individuo = None
        self.mejor_fitness = -np.inf
        self.historial_fitness = []
        
    def crear_individuo_aleatorio(self):
        """Crear un cromosoma aleatorio"""
        num_capas_ocultas = random.randint(1, self.max_capas)
        
        # Crear lista de neuronas por capa oculta
        neuronas = []
        for _ in range(num_capas_ocultas):
            neuronas.append(random.randint(2, self.max_neuronas))
        
        # Agregar índice de tasa de aprendizaje
        tasa_lr_index = random.randint(0, len(self.tasas_aprendizaje) - 1)
        
        return {
            'capas_ocultas': neuronas,
            'tasa_lr_index': tasa_lr_index
        }
    
    def crear_poblacion_inicial(self):
        """Crear población inicial aleatoria"""
        return [self.crear_individuo_aleatorio() for _ in range(self.poblacion_size)]
    
    def cromosoma_a_arquitectura(self, cromosoma):
        """Convertir cromosoma a arquitectura de red neuronal"""
        capas_ocultas = cromosoma['capas_ocultas']
        arquitectura = [1] + capas_ocultas + [1]  # 1 entrada, capas ocultas, 1 salida
        tasa_aprendizaje = self.tasas_aprendizaje[cromosoma['tasa_lr_index']]
        
        return arquitectura, tasa_aprendizaje
    
    def evaluar_fitness(self, cromosoma, X_train, y_train, X_val, y_val):
        """Evaluar fitness de un cromosoma"""
        try:
            arquitectura, tasa_aprendizaje = self.cromosoma_a_arquitectura(cromosoma)
            
            # Crear y entrenar red neuronal
            red = RedNeuronalMulticapa(arquitectura, tasa_aprendizaje)
            red.entrenar(X_train, y_train, epocas=self.epocas_entrenamiento, mostrar_progreso=False)
            
            # Evaluar en conjunto de validación
            metricas = red.evaluar(X_val, y_val)
            
            # El fitness es el R² score (queremos maximizar)
            # Penalizar redes muy grandes para evitar sobreajuste
            num_parametros = sum(arquitectura[i] * arquitectura[i+1] for i in range(len(arquitectura)-1))
            penalizacion = num_parametros * 0.0001  # Penalización pequeña
            
            fitness = metricas['R²'] - penalizacion
            
            return fitness
            
        except Exception as e:
            # Si hay error, fitness muy bajo
            return -1.0
    
    def seleccion_torneo(self, poblacion, fitness_scores, k=3):
        """Selección por torneo"""
        seleccionados = []
        
        for _ in range(len(poblacion)):
            # Seleccionar k individuos aleatorios
            competidores_indices = random.sample(range(len(poblacion)), k)
            
            # Encontrar el mejor de los k
            mejor_index = max(competidores_indices, key=lambda i: fitness_scores[i])
            seleccionados.append(poblacion[mejor_index].copy())
        
        return seleccionados
    
    def cruce(self, padre1, padre2):
        """Cruce de dos cromosomas"""
        if random.random() > self.prob_cruce:
            return padre1.copy(), padre2.copy()
        
        # Cruce en capas ocultas
        capas1 = padre1['capas_ocultas'].copy()
        capas2 = padre2['capas_ocultas'].copy()
        
        # Punto de cruce aleatorio
        if len(capas1) > 1 and len(capas2) > 1:
            punto1 = random.randint(1, len(capas1))
            punto2 = random.randint(1, len(capas2))
            
            # Intercambiar segmentos
            nueva_capas1 = capas1[:punto1] + capas2[punto2:]
            nueva_capas2 = capas2[:punto2] + capas1[punto1:]
        else:
            nueva_capas1 = capas1
            nueva_capas2 = capas2
        
        # Cruce en tasa de aprendizaje (50% de probabilidad)
        if random.random() < 0.5:
            tasa1 = padre2['tasa_lr_index']
            tasa2 = padre1['tasa_lr_index']
        else:
            tasa1 = padre1['tasa_lr_index']
            tasa2 = padre2['tasa_lr_index']
        
        hijo1 = {'capas_ocultas': nueva_capas1, 'tasa_lr_index': tasa1}
        hijo2 = {'capas_ocultas': nueva_capas2, 'tasa_lr_index': tasa2}
        
        return hijo1, hijo2
    
    def mutacion(self, cromosoma):
        """Mutar un cromosoma"""
        if random.random() > self.prob_mutacion:
            return cromosoma
        
        cromosoma_mutado = cromosoma.copy()
        
        # Tipos de mutación
        tipo_mutacion = random.choice(['agregar_capa', 'quitar_capa', 'cambiar_neuronas', 'cambiar_tasa'])
        
        if tipo_mutacion == 'agregar_capa' and len(cromosoma_mutado['capas_ocultas']) < self.max_capas:
            nueva_capa = random.randint(2, self.max_neuronas)
            posicion = random.randint(0, len(cromosoma_mutado['capas_ocultas']))
            cromosoma_mutado['capas_ocultas'].insert(posicion, nueva_capa)
            
        elif tipo_mutacion == 'quitar_capa' and len(cromosoma_mutado['capas_ocultas']) > 1:
            posicion = random.randint(0, len(cromosoma_mutado['capas_ocultas']) - 1)
            cromosoma_mutado['capas_ocultas'].pop(posicion)
            
        elif tipo_mutacion == 'cambiar_neuronas':
            if cromosoma_mutado['capas_ocultas']:
                posicion = random.randint(0, len(cromosoma_mutado['capas_ocultas']) - 1)
                cromosoma_mutado['capas_ocultas'][posicion] = random.randint(2, self.max_neuronas)
                
        elif tipo_mutacion == 'cambiar_tasa':
            cromosoma_mutado['tasa_lr_index'] = random.randint(0, len(self.tasas_aprendizaje) - 1)
        
        return cromosoma_mutado
    
    def evolucionar(self, X_train, y_train, X_val, y_val):
        """Proceso principal del algoritmo genético"""
        print("\n INICIANDO ALGORITMO GENÉTICO")
        print("=" * 50)
        print(f"Población: {self.poblacion_size} individuos")
        print(f"Generaciones: {self.generaciones}")
        print(f"Prob. Mutación: {self.prob_mutacion}")
        print(f"Prob. Cruce: {self.prob_cruce}")
        print("-" * 50)
        
        # Crear población inicial
        poblacion = self.crear_poblacion_inicial()
        
        for generacion in range(self.generaciones):
            print(f"\n Generación {generacion + 1}/{self.generaciones}")
            
            # Evaluar fitness de toda la población
            fitness_scores = []
            for i, individuo in enumerate(poblacion):
                fitness = self.evaluar_fitness(individuo, X_train, y_train, X_val, y_val)
                fitness_scores.append(fitness)
                print(f"  Individuo {i+1:2d}: Fitness = {fitness:.4f} | Arquitectura = {self.cromosoma_a_arquitectura(individuo)}")
            
            # Encontrar el mejor individuo de esta generación
            mejor_index_gen = np.argmax(fitness_scores)
            mejor_fitness_gen = fitness_scores[mejor_index_gen]
            
            # Actualizar mejor global
            if mejor_fitness_gen > self.mejor_fitness:
                self.mejor_fitness = mejor_fitness_gen
                self.mejor_individuo = poblacion[mejor_index_gen].copy()
            
            self.historial_fitness.append(mejor_fitness_gen)
            
            print(f"   Mejor de la generación: Fitness = {mejor_fitness_gen:.4f}")
            print(f"   Mejor global: Fitness = {self.mejor_fitness:.4f}")
            
            # Si no es la última generación, crear nueva población
            if generacion < self.generaciones - 1:
                # Selección
                nueva_poblacion = self.seleccion_torneo(poblacion, fitness_scores)
                
                # Cruce y mutación
                poblacion_cruzada = []
                for i in range(0, len(nueva_poblacion), 2):
                    if i + 1 < len(nueva_poblacion):
                        hijo1, hijo2 = self.cruce(nueva_poblacion[i], nueva_poblacion[i + 1])
                        poblacion_cruzada.extend([hijo1, hijo2])
                    else:
                        poblacion_cruzada.append(nueva_poblacion[i])
                
                # Mutación
                poblacion = [self.mutacion(individuo) for individuo in poblacion_cruzada]
                
                # Elitismo: mantener el mejor individuo
                poblacion[0] = self.mejor_individuo.copy()
        
        print(f"\n ALGORITMO GENÉTICO COMPLETADO")
        print(f" MEJOR CONFIGURACIÓN ENCONTRADA:")
        arquitectura, tasa_lr = self.cromosoma_a_arquitectura(self.mejor_individuo)
        print(f"   Arquitectura: {arquitectura}")
        print(f"   Tasa de aprendizaje: {tasa_lr}")
        print(f"   Fitness (R²): {self.mejor_fitness:.6f}")
        
        return self.mejor_individuo


def cargar_datos(nombre_archivo='puntos.txt'):
    """Cargar datos desde archivo"""
    print(f" Cargando datos desde {nombre_archivo}...")
    
    try:
        datos = []
        with open(nombre_archivo, 'r') as archivo:
            for linea in archivo:
                linea_limpia = linea.strip().replace('(', '').replace(')', '')
                x, y = map(float, linea_limpia.split(','))
                datos.append([x, y])
        
        datos = np.array(datos)
        X = datos[:, 0].reshape(-1, 1)
        y = datos[:, 1].reshape(-1, 1)
        
        print(f"   Datos cargados: {len(datos)} puntos")
        print(f"   Rango X: [{X.min():.2f}, {X.max():.2f}]")
        print(f"   Rango Y: [{y.min():.2f}, {y.max():.2f}]")
        
        return X, y
        
    except FileNotFoundError:
        print(f" Error: No se encontró el archivo '{nombre_archivo}'")
        return None, None
    except Exception as e:
        print(f" Error cargando datos: {e}")
        return None, None


def visualizar_evolucion_ag(historial_fitness):
    """Visualizar evolución del algoritmo genético"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(historial_fitness) + 1), historial_fitness, 'b-', linewidth=2, marker='o')
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fitness (R²)')
    plt.title('Evolución del Algoritmo Genético')
    plt.grid(True, alpha=0.3)
    plt.show()


def visualizar_resultados_finales(X, y, y_pred, metricas, arquitectura):
    """Visualizar resultados del mejor modelo"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico 1: Datos vs Predicciones
    indices_ordenados = np.argsort(X.flatten())
    ax1.scatter(X, y, alpha=0.6, color='blue', s=30, label='Datos Reales')
    ax1.plot(X[indices_ordenados], y_pred[indices_ordenados], color='red', linewidth=2, label='Predicciones')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Mejor Modelo: Arquitectura {arquitectura}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Residuos
    residuos = y - y_pred
    ax2.scatter(y_pred, residuos, alpha=0.6, color='green', s=30)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Predicciones')
    ax2.set_ylabel('Residuos')
    ax2.set_title('Análisis de Residuos')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Valores Reales vs Predicciones
    ax3.scatter(y, y_pred, alpha=0.6, color='purple', s=30)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax3.set_xlabel('Valores Reales')
    ax3.set_ylabel('Predicciones')
    ax3.set_title('Correlación Real vs Predicción')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Métricas
    metricas_nombres = list(metricas.keys())
    metricas_valores = list(metricas.values())
    colores = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = ax4.bar(metricas_nombres, metricas_valores, color=colores)
    ax4.set_title('Métricas de Evaluación')
    ax4.set_ylabel('Valor')
    
    for bar, valor in zip(bars, metricas_valores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{valor:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def main():
    """Función principal"""
    print("=" * 70)
    print(" OPTIMIZACIÓN DE RED NEURONAL CON ALGORITMOS GENÉTICOS")
    print("=" * 70)
    
    # 1. Cargar datos
    X, y = cargar_datos('puntos.txt')
    if X is None:
        return
    
    # 2. Dividir datos: entrenamiento, validación y prueba
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"\n División de datos:")
    print(f"   Entrenamiento: {len(X_train)} muestras")
    print(f"   Validación: {len(X_val)} muestras")
    print(f"   Prueba: {len(X_test)} muestras")
    
    # 3. Normalizar datos
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(" Datos normalizados")
    
    # 4. Configurar y ejecutar algoritmo genético
    ag = AlgoritmoGenetico(
        poblacion_size=35,      # Población más pequeña para rapidez
        generaciones=15,         # Pocas generaciones para demostración
        prob_mutacion=0.15,     # Probabilidad de mutación
        prob_cruce=0.8,         # Probabilidad de cruce
        max_capas=5,            # Máximo 3 capas ocultas
        max_neuronas=20,        # Máximo 20 neuronas por capa
        epocas_entrenamiento=400  # Pocas épocas para rapidez
    )
    
    # Ejecutar algoritmo genético
    mejor_cromosoma = ag.evolucionar(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
    
    # 5. Visualizar evolución del AG
    print(f"\n Visualizando evolución del algoritmo genético...")
    visualizar_evolucion_ag(ag.historial_fitness)
    
    # 6. Entrenar modelo final con mejor configuración
    print(f"\n Entrenando modelo final con mejor configuración...")
    arquitectura_final, tasa_lr_final = ag.cromosoma_a_arquitectura(mejor_cromosoma)
    
    modelo_final = RedNeuronalMulticapa(arquitectura_final, tasa_lr_final)
    modelo_final.entrenar(X_train_scaled, y_train_scaled, epocas=800, mostrar_progreso=True)
    
    # 7. Evaluar en conjunto de prueba
    y_pred_test_scaled = modelo_final.predecir(X_test_scaled)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)
    
    # Métricas finales
    mse_final = mean_squared_error(y_test, y_pred_test)
    r2_final = r2_score(y_test, y_pred_test)
    rmse_final = np.sqrt(mse_final)
    
    metricas_finales = {'MSE': mse_final, 'RMSE': rmse_final, 'R²': r2_final}
    
    print(f"\n RESULTADOS FINALES:")
    print(f"   Arquitectura optimizada: {arquitectura_final}")
    print(f"   Tasa de aprendizaje: {tasa_lr_final}")
    print(f"    Métricas en conjunto de prueba:")
    for metrica, valor in metricas_finales.items():
        print(f"      {metrica}: {valor:.6f}")
    
    # 8. Visualizar resultados finales
    print(f"\n Generando visualizaciones finales...")
    visualizar_resultados_finales(X_test, y_test, y_pred_test, metricas_finales, arquitectura_final)
    
    # 9. Análisis de resultados
    print(f"\n ANÁLISIS DE RESULTADOS:")
    print(f"   • R² = {metricas_finales['R²']:.4f}: ", end="")
    if metricas_finales['R²'] > 0.8:
        print("¡Excelente ajuste!")
    elif metricas_finales['R²'] > 0.6:
        print("Buen ajuste")
    elif metricas_finales['R²'] > 0.4:
        print("Ajuste moderado")
    else:
        print("Ajuste pobre")
    
    print(f"   • RMSE = {metricas_finales['RMSE']:.4f}: Error promedio")
    print(f"   • El algoritmo genético encontró una arquitectura optimizada automáticamente")
    
    print(f"\n ¡Optimización completada exitosamente!")


if __name__ == "__main__":
    main()
