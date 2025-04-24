import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

class Control:
    def __init__(self):
        pos_max = np.pi
        vel_max = 3
        f_max = 50

        # Variables linguisticas
        self.posicion = ctrl.Antecedent(np.linspace(-pos_max, pos_max, 10000), 'posicion')
        self.velocidad = ctrl.Antecedent(np.linspace(-vel_max, vel_max, 10000), 'velocidad')
        self.fuerza = ctrl.Consequent(np.linspace(-f_max, f_max, 10000), 'fuerza')

        # Definicion de conjuntos borrosos
        self.posicion['NG'] = fuzz.trapmf(self.posicion.universe, np.array([-pos_max, -pos_max, -16 * np.pi / 180, -7 * np.pi / 180]))
        self.posicion['NP'] = fuzz.trimf(self.posicion.universe, np.array([-16 * np.pi / 180, -9 * np.pi / 180, 0]))
        self.posicion['Z'] = fuzz.trimf(self.posicion.universe, np.array([-7 * np.pi / 180, 0, 7 * np.pi / 180]))
        self.posicion['PP'] = fuzz.trimf(self.posicion.universe, np.array([0, 9 * np.pi / 180, 16 * np.pi / 180]))
        self.posicion['PG'] = fuzz.trapmf(self.posicion.universe, np.array([7 * np.pi / 180, 16 * np.pi / 180, pos_max, pos_max]))

        self.velocidad['NG'] = fuzz.trapmf(self.velocidad.universe, np.array([-vel_max, -vel_max, -2, -1]))
        self.velocidad['NP'] = fuzz.trimf(self.velocidad.universe, np.array([-2, -1, 0]))
        self.velocidad['Z'] = fuzz.trimf(self.velocidad.universe, np.array([-1, 0, 1]))
        self.velocidad['PP'] = fuzz.trimf(self.velocidad.universe, np.array([0, 1, 2]))
        self.velocidad['PG'] = fuzz.trapmf(self.velocidad.universe, np.array([1, 2, vel_max, vel_max]))

        self.fuerza['NG'] = fuzz.trapmf(self.fuerza.universe, np.array([-f_max, -f_max, -25, -10]))
        self.fuerza['NP'] = fuzz.trimf(self.fuerza.universe, np.array([-20, -10, 0]))
        self.fuerza['Z'] = fuzz.trimf(self.fuerza.universe, np.array([-10, 0, 10]))
        self.fuerza['PP'] = fuzz.trimf(self.fuerza.universe, np.array([0, 10, 20]))
        self.fuerza['PG'] = fuzz.trapmf(self.fuerza.universe, np.array([10, 25, f_max, f_max]))

        self.fuerza.defuzzify_method = 'centroid'
        # Por media del valor maximo
        #self.fuerza.defuzzify_method = 'mom'
        # Por absisa minima del valor maximo
        #self.fuerza.defuzzify_method = 'som'
        # Por absisa maxima del valor maximo
        #self.fuerza.defuzzify_method = 'lom'
        # Por punto que divide el area en 2 partes iguales
        #self.fuerza.defuzzify_method = 'bisector'

        # Base de conocimiento
        etiquetas = ['NG', 'NP', 'Z', 'PP', 'PG']

        # Matriz de salida (fuerza) como lista de listas (posición x velocidad)
        matriz_fuerza = [
            ['PG', 'PG', 'PG', 'PP', 'Z'],
            ['PG', 'PG', 'PP', 'Z' , 'Z'],
            ['PP', 'Z' , 'Z' , 'Z' , 'NG'],
            ['Z' , 'Z' , 'NP', 'NG', 'NG'],
            ['Z' , 'NP', 'NG', 'NG', 'NG']
        ]

        """
        matriz_fuerza = [
            ['PG', 'PG', 'PG', 'PG', 'PP'],
            ['PG', 'PG', 'PG', 'PP', 'PP'],
            ['PG', 'PP', 'Z', 'NP', 'NG'],
            ['NP', 'NP', 'NP', 'NG', 'NG'],
            ['NP', 'NG', 'NG', 'NG', 'NG']
        ]
        """

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
        plt.show()

class Carro:
    def __init__(self):
        # Definir posición y velocidad inicial
        self.posicion = 10 * np.pi / 180
        self.velocidad = -1.5
        self.aceleracion = 0

        self.a = 0.0
        self.posicion_carro = [0.0]
        self.velocidad_carro = [0.0]

        # Masa carro y pendulo
        self.m = 0.1
        self.M = 1

        # Longitud pendulo
        self.l = 0.3

        # Intervalo entre iteraciones
        self.dt = 0.01

    def calc_aceleracion(self, fuerza):
        self.a = (((4/3)*(fuerza + self.m * self.l * self.velocidad**2 * np.sin(self.posicion)) - (self.m * 9.81 * np.sin(self.posicion) * np.cos(self.posicion))) / ((4/3) * (self.M + self.m) - self.m * np.cos(self.posicion)**2))
        self.aceleracion = (9.81 * np.sin(self.posicion) + np.cos(self.posicion) * ((fuerza - self.m * self.l * self.velocidad**2 * np.sin(self.posicion)) / (self.M + self.m))) / (self.l * ((4/3) - (self.m * (np.cos(self.posicion))**2) / (self.M + self.m)))

    def calc_velocidad(self):
        self.velocidad_carro.append(self.velocidad_carro[-1] + self.a * self.dt)
        self.velocidad += self.aceleracion * self.dt

    def calc_posicion(self):
        self.posicion_carro.append(self.posicion_carro[-1] + self.velocidad_carro[-1] * self.dt + 0.5 * self.a * self.dt**2)
        temp = self.posicion + self.velocidad * self.dt + 0.5 * self.aceleracion * self.dt**2
        if temp < -np.pi:
            temp += 2 * np.pi
        elif temp > np.pi:
            temp -= 2 * np.pi
        self.posicion = temp

    def simular(self, fuerza):
        self.calc_aceleracion(fuerza)
        self.calc_velocidad()
        self.calc_posicion()

class Simulador:
    def __init__(self):
        self.control = Control()
        self.carro = Carro()
        self.fuerza = [0]
        self.running = True
        self.posiciones = [self.carro.posicion]
        self.velocidades = [self.carro.velocidad]
        self.tiempo_simulacion = 5

    def thread_fisico(self):
        self.carro.simular(self.fuerza[-1])

    def thread_control(self):
        posicion = self.carro.posicion
        self.posiciones.append(posicion)
        velocidad = self.carro.velocidad
        self.velocidades.append(velocidad)
        self.fuerza.append(self.control.calcular(posicion, velocidad))

    def graficar(self):
        t = np.linspace(0, self.tiempo_simulacion, int(self.tiempo_simulacion / self.carro.dt) + 1)

        plt.figure(1)
        plt.plot(t, self.posiciones, label='Posición')
        plt.plot(t, self.velocidades, label='Velocidad')
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Valor")
        plt.title("Evolución del sistema")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(2)
        plt.plot(t, self.carro.posicion_carro, label='Posición carro')
        plt.plot(t, self.carro.velocidad_carro, label='Velocidad carro')
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(3)
        plt.plot(t, self.fuerza, label='Fuerza')
        plt.legend()
        plt.grid()
        plt.show()

    def run(self):
        self.control.ver_conjuntos_borrosos()
        for tiempo in range(0, int(self.tiempo_simulacion / self.carro.dt)):
            self.thread_control()
            self.thread_fisico()

        self.running = False

        self.graficar()

def debug():
    control = Control()
    control.ver_conjuntos_borrosos()

def main():
    simulador = Simulador()
    simulador.run()

if __name__ == '__main__':
    main()