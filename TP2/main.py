import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import time
import threading

class Control:
    def __init__(self):
        # Variables linguisticas
        self.posicion = ctrl.Antecedent(np.linspace(-np.pi, np.pi, 10000), 'posicion')
        self.velocidad = ctrl.Antecedent(np.linspace(-3, 3, 10000), 'velocidad')
        self.fuerza = ctrl.Consequent(np.linspace(-10, 10, 10000), 'fuerza')

        # Definicion de conjuntos borrosos
        self.posicion['NG'] = fuzz.zmf(self.posicion.universe, -45 * np.pi / 180, -20*np.pi/180)
        self.posicion['NP'] = fuzz.trimf(self.posicion.universe, np.array([-30 * np.pi / 180, -17.5 * np.pi / 180, -5 * np.pi / 180]))
        self.posicion['Z'] = fuzz.trimf(self.posicion.universe, np.array([-10 * np.pi / 180, 0, 10 * np.pi / 180]))
        self.posicion['PP'] = fuzz.trimf(self.posicion.universe, np.array([5 * np.pi / 180, 17.5 * np.pi / 180, 30 * np.pi / 180]))
        self.posicion['PG'] = fuzz.smf(self.posicion.universe, 20 * np.pi / 180, 45 * np.pi / 180)

        self.velocidad['NG'] = fuzz.zmf(self.velocidad.universe, -2.5, -1.5)
        self.velocidad['NP'] = fuzz.trimf(self.velocidad.universe, np.array([-2, -1.25, -0.5]))
        self.velocidad['Z'] = fuzz.trimf(self.velocidad.universe, np.array([-1, 0, 1]))
        self.velocidad['PP'] = fuzz.trimf(self.velocidad.universe, np.array([0.5, 1.25, 2]))
        self.velocidad['PG'] = fuzz.smf(self.velocidad.universe, 1.5, 2.5)

        self.fuerza['NG'] = fuzz.zmf(self.fuerza.universe, -4, -2.5)
        self.fuerza['NP'] = fuzz.trimf(self.fuerza.universe, np.array([-3, -1.75, -0.5]))
        self.fuerza['Z'] = fuzz.trimf(self.fuerza.universe, np.array([-1, 0, 1]))
        self.fuerza['PP'] = fuzz.trimf(self.fuerza.universe, np.array([0.5, 1.75, 3]))
        self.fuerza['PG'] = fuzz.smf(self.fuerza.universe, 2.5, 4)

        # Base de conocimiento
        etiquetas = ['NG', 'NP', 'Z', 'PP', 'PG']

        # Matriz de salida (fuerza) como lista de listas (posición x velocidad)
        matriz_fuerza = [
            ['PG', 'PG', 'PP', 'PG', 'PG'],
            ['PG', 'PP', 'PP', 'PP', 'PG'],
            ['PG', 'PP', 'Z', 'NP', 'NG'],
            ['NG', 'NP', 'NP', 'NP', 'NG'],
            ['NG', 'NG', 'NG', 'NG', 'NG']
        ]

        self.reglas = []
        for i, pos_label in enumerate(etiquetas):
            for j, vel_label in enumerate(etiquetas):
                salida_label = matriz_fuerza[i][j]
                regla = ctrl.Rule(antecedent=(self.posicion[pos_label] & self.velocidad[vel_label]),
                                  consequent=self.fuerza[salida_label],
                                  label=f"R_{pos_label}_{vel_label}")
                self.reglas.append(regla)

        # Creo y simulo sistema de control
        self.sistema_control = ctrl.ControlSystem(self.reglas)
        self.simulador = ctrl.ControlSystemSimulation(self.sistema_control)

    def calcular(self, lec_pos, lec_vel):
        self.simulador.input['posicion'] = lec_pos
        self.simulador.input['velocidad'] = lec_vel

        self.simulador.compute()

        return self.simulador.output['fuerza']

    def ver_conjuntos_borrosos(self):
        self.posicion.view()
        self.velocidad.view()
        self.fuerza.view()
        print(self.reglas)
        plt.show()

class Carro:
    def __init__(self):
        # Definir posición y velocidad inicial
        self.posicion = -0.1
        self.velocidad = 0.5
        self.aceleracion = 0

        # Masa carro y pendulo
        self.m = 0.1
        self.M = 1

        # Longitud pendulo
        self.l = 0.3

        # Intervalo entre iteraciones
        self.dt = 0.01

    def calc_aceleracion(self, fuerza):
        self.aceleracion = (9.81 * np.sin(self.posicion) + np.cos(self.posicion) * ((-fuerza - self.m * self.l * self.velocidad**2 * np.sin(self.posicion))/(self.M + self.m))) / (self.l * ((4/3) - (self.m * np.cos(self.posicion)**2) / (self.M + self.m)))

    def calc_velocidad(self):
        self.velocidad = self.velocidad + self.aceleracion * self.dt

    def calc_posicion(self):
        self.posicion = self.posicion + self.velocidad * self.dt + 0.5 * self.aceleracion * self.dt**2

    def simular(self, fuerza):
        self.calc_aceleracion(fuerza)
        self.calc_velocidad()
        self.calc_posicion()

class Simulador:
    def __init__(self):
        self.control = Control()
        self.carro = Carro()
        self.fuerza = 0
        self.running = True
        self.lock = threading.Lock()
        self.posiciones = []
        self.velocidades = []

    def thread_fisico(self):
        self.carro.simular(self.fuerza)
        #print(f"Fisico: {time.time()}")

    def thread_control(self):
        posicion = self.carro.posicion
        self.posiciones.append(posicion)
        velocidad = self.carro.velocidad
        self.velocidades.append(velocidad)
        self.fuerza = self.control.calcular(posicion, velocidad)
        #print(f"Control: {time.time()}")

    def run(self, tiempo_simulacion=10):
        for tiempo in range(0, int(tiempo_simulacion / self.carro.dt)):
            self.thread_control()
            self.thread_fisico()

        self.running = False

        t = np.linspace(0, tiempo_simulacion, int(tiempo_simulacion / self.carro.dt))

        plt.plot(t, self.posiciones, label='Posición')
        plt.plot(t, self.velocidades, label='Velocidad')
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Valor")
        plt.title("Evolución del sistema")
        plt.legend()
        plt.grid()
        plt.show()

def main():
    simulador = Simulador()
    simulador.run()

if __name__ == '__main__':
    main()