import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


class Control:
    def __init__(self):
        # Variables linguisticas
        self.posicion = ctrl.Antecedent(np.linspace(-np.pi, np.pi, 10000), 'posicion')
        self.velocidad = ctrl.Antecedent(np.linspace(-3, 3, 1000), 'velocidad')
        self.fuerza = ctrl.Consequent(np.linspace(-10, 10, 1000), 'fuerza')

        # Definicion de conjuntos borrosos
        self.posicion['NG'] = fuzz.zmf(self.posicion.universe, -45 * np.pi / 180, -20*np.pi/180)
        self.posicion['NP'] = fuzz.trimf(self.posicion.universe, np.array([-30 * np.pi / 180, -17.5 * np.pi / 180, -5 * np.pi / 180]))
        self.posicion['Z'] = fuzz.trimf(self.posicion.universe, np.array([-10 * np.pi / 180, 0, 10 * np.pi / 180]))
        self.posicion['PP'] = fuzz.trimf(self.posicion.universe, np.array([5 * np.pi / 180, 17.5 * np.pi / 180, 30 * np.pi / 180]))
        self.posicion['PG'] = fuzz.smf(self.posicion.universe, 20 * np.pi / 180, 45 * np.pi / 180)

    def ver_conjuntos_borrosos(self):
        self.posicion.view()
        plt.show()



if __name__ == '__main__':
    control = Control()
    control.ver_conjuntos_borrosos()