import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Parámetros físicos (SI)
g = 9.81     # gravedad (m/s^2)
l = 0.1      # longitud del péndulo (m)
m = 0.1      # masa del péndulo (kg)
M = 1.0      # masa del carro (kg)
dt = 0.001   # paso de integración (s)

# Condición inicial para simulación
theta = np.radians(30)   # rad
theta_dot = -1.5            # rad/s
tiempo_simulado = 1      # tiempo de simulación (s)

# Máximos de universo en radianes y rad/s
angulo_max = np.radians(90)  # rad (90°)
velocidad_max = 20           # rad/s
fuerza_max = 200            # Newtons

# Crear universos para las variables de entrada (ángulo y velocidad) y salida (fuerza)
angulo = ctrl.Antecedent(np.linspace(-angulo_max, angulo_max, 1001), 'angulo')
velocidad = ctrl.Antecedent(np.linspace(-velocidad_max, velocidad_max, 1001), 'velocidad')

fuerza = ctrl.Consequent(np.linspace(-fuerza_max, fuerza_max, 1001), 'fuerza')
#fuerza.defuzzify_method = 'centroid'
# Por media del valor maximo
#fuerza.defuzzify_method = 'mom'
# Por absisa minima del valor maximo
#fuerza.defuzzify_method = 'som'
# Por absisa maxima del valor maximo
#fuerza.defuzzify_method = 'lom'
# Por punto que divide el area en 2 partes iguales
fuerza.defuzzify_method = 'bisector'

# Función para definir los centros de las funciones de pertenencia
def centros(maximo):
    return [-0.5*maximo, -0.25*maximo, 0, 0.25*maximo, 0.5*maximo]

# Función para generar las funciones de pertenencia
def generar_mfs(var, maximo, angosto_Z=False):
    c = centros(maximo)
    var['NG'] = fuzz.trapmf(var.universe, [var.universe[0], var.universe[0], c[0], c[1]])  # Conjunto NG (Negative Big)
    var['NP'] = fuzz.trimf(var.universe, [c[0], c[1], c[2]])  # Conjunto NP (Negative Small)
    if angosto_Z:
        var['Z'] = fuzz.trimf(var.universe, [c[1]*0.5, 0, c[3]*0.5])  # Función 'Z' más angosta
    else:
        var['Z'] = fuzz.trimf(var.universe, [c[1], c[2], c[3]])  # Conjunto Z (Zero)
    var['PP'] = fuzz.trimf(var.universe, [c[2], c[3], c[4]])  # Conjunto PP (Positive Small)
    var['PG'] = fuzz.trapmf(var.universe, [c[3], c[4], var.universe[-1], var.universe[-1]])  # Conjunto PG (Positive Big)

# Generar funciones de pertenencia para ángulo, velocidad y fuerza
generar_mfs(angulo, angulo_max, angosto_Z=0)  # Funciones de pertenencia para ángulo
generar_mfs(velocidad, velocidad_max, angosto_Z=0)  # Funciones de pertenencia para velocidad
generar_mfs(fuerza, fuerza_max, angosto_Z=0)  # Funciones de pertenencia para fuerza

# Reglas lógicas para control difuso del péndulo
rules = [
    ctrl.Rule(angulo['PG'] & velocidad['PG'], fuerza['NG']),
    ctrl.Rule(angulo['PG'] & velocidad['PP'], fuerza['NG']),
    ctrl.Rule(angulo['PG'] & velocidad['Z'],  fuerza['NG']),
    ctrl.Rule(angulo['PG'] & velocidad['NP'], fuerza['NP']),
    ctrl.Rule(angulo['PG'] & velocidad['NG'], fuerza['Z']),  # Freno inteligente
    
    ctrl.Rule(angulo['PP'] & velocidad['PG'], fuerza['NG']),
    ctrl.Rule(angulo['PP'] & velocidad['PP'], fuerza['NG']),
    ctrl.Rule(angulo['PP'] & velocidad['Z'],  fuerza['NP']),
    ctrl.Rule(angulo['PP'] & velocidad['NP'], fuerza['Z']),
    ctrl.Rule(angulo['PP'] & velocidad['NG'], fuerza['Z']), 
    
    ctrl.Rule(angulo['Z'] & velocidad['PG'], fuerza['NP']),
    ctrl.Rule(angulo['Z'] & velocidad['PP'], fuerza['Z']),
    ctrl.Rule(angulo['Z'] & velocidad['Z'],  fuerza['Z']),
    ctrl.Rule(angulo['Z'] & velocidad['NP'], fuerza['Z']),
    ctrl.Rule(angulo['Z'] & velocidad['NG'], fuerza['PP']),
    
    ctrl.Rule(angulo['NP'] & velocidad['PG'], fuerza['Z']),
    ctrl.Rule(angulo['NP'] & velocidad['PP'], fuerza['Z']),
    ctrl.Rule(angulo['NP'] & velocidad['Z'],  fuerza['PP']),
    ctrl.Rule(angulo['NP'] & velocidad['NP'], fuerza['PG']),
    ctrl.Rule(angulo['NP'] & velocidad['NG'], fuerza['PG']),
    
    ctrl.Rule(angulo['NG'] & velocidad['PG'], fuerza['Z']),  # Freno inteligente
    ctrl.Rule(angulo['NG'] & velocidad['PP'], fuerza['PP']),
    ctrl.Rule(angulo['NG'] & velocidad['Z'],  fuerza['PG']),
    ctrl.Rule(angulo['NG'] & velocidad['NP'], fuerza['PG']),
    ctrl.Rule(angulo['NG'] & velocidad['NG'], fuerza['PG']),
]

# Crear sistema y simulador
sistema = ctrl.ControlSystem(rules)
simulador = ctrl.ControlSystemSimulation(sistema)

# Fuzzificación de entradas: convertir entradas nítidas (ángulo y velocidad) en conjuntos difusos.
simulador.input['angulo'] = theta
simulador.input['velocidad'] = theta_dot
simulador.compute()

# Fuzzificación: convertimos las entradas nítidas (valor numérico) en grados de pertenencia borrosos.

# Graficar las particiones borrosas de entrada (ángulo)
# Distribución de las funciones de pertenencia de la variable 'ángulo'.
angulo.view(sim=simulador)
plt.title("Particiones borrosas del Ángulo inicial")
plt.xlabel("Ángulo (rad)") 
plt.ylabel("Grado de pertenencia")  
plt.show()

# Graficar las particiones borrosas de entrada (velocidad)
velocidad.view(sim=simulador)
plt.title("Particiones borrosas de la Velocidad inicial")
plt.xlabel("Velocidad (rad/s)") 
plt.ylabel("Grado de pertenencia")  
plt.show()

# Mostrar gráfico de salida borrosa (Fuerza) con la combinación de los conjuntos borrosos
# La combinación de conjuntos borrosos de salida ocurre en el motor de inferencia y se refleja en el gráfico final.
# Resaltamos cómo se combinan los conjuntos borrosos (por ejemplo, Z y PP) en la salida final.

# Graficar la salida combinada de los conjuntos borrosos
fuerza.view(sim=simulador)  # Mostramos la combinación de las funciones de pertenencia de la salida
plt.title("Combinación de Conjuntos Borrosos de la Fuerza")
plt.xlabel("Fuerza (N)") 
plt.ylabel("Grado de pertenencia")
plt.show()

# Simulación
tiempo, angulos, velocidades, fuerzas = [], [], [], []

# Proceso de simulación utilizando el modelo dinámico y las reglas definidas
for t in np.arange(0, tiempo_simulado, dt):
    simulador.input['angulo'] = theta
    simulador.input['velocidad'] = theta_dot
    simulador.compute()
    
    F = simulador.output['fuerza'] if 'fuerza' in simulador.output else 0.0
    
    # Guardar datos en el arreglo previamente definino para luego graficar
    tiempo.append(t)
    angulos.append(theta)
    velocidades.append(theta_dot)
    fuerzas.append(F)

    # Actualización del modelo dinámico con la aceleración angular
    theta_ddot = (g * np.sin(theta) + np.cos(theta) * ((F - m * l * theta_dot**2 * np.sin(theta)) / (M + m))) / (l * (4/3 - (m * np.cos(theta)**2) / (M + m)))

    # Integración por método numérico para actualizar la velocidad y el ángulo
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt + 0.5 * theta_ddot * dt**2

# Graficar resultados de la simulación
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(tiempo, angulos, color='red')
plt.ylabel("Posición (rad)")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(tiempo, velocidades, color='yellow')
plt.ylabel("Velocidad (rad/seg)")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(tiempo, fuerzas, color='green')
plt.xlabel("Tiempo (seg)")
plt.ylabel("Fuerza (N)")
plt.grid()

plt.tight_layout()
plt.show()
