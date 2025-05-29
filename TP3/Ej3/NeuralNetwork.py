import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.initialize()

    def initialize(self):
        # ======================== INITIALIZE NETWORK WEIGHTS AND BIASES =============================
        
        # Definir la arquitectura de la red (aumentada para mejor aprendizaje)
        self.input_size = 5      # Entradas: distancia_x, distancia_y, velocidad_juego, altura_dinosaurio, tipo_obstaculo
        self.hidden_size1 = 24   # Primera capa oculta (más neuronas)
        self.hidden_size2 = 16   # Segunda capa oculta (arquitectura más profunda)
        self.output_size = 3     # Salidas: JUMP, DUCK, RUN
        
        # Inicialización Xavier/Glorot para mejor convergencia
        # Pesos entre capa de entrada y primera capa oculta
        self.weights_input_hidden1 = np.random.normal(0, np.sqrt(2.0 / self.input_size), 
                                                      (self.input_size, self.hidden_size1))
        
        # Pesos entre primera y segunda capa oculta
        self.weights_hidden1_hidden2 = np.random.normal(0, np.sqrt(2.0 / self.hidden_size1), 
                                                        (self.hidden_size1, self.hidden_size2))
        
        # Pesos entre segunda capa oculta y capa de salida  
        self.weights_hidden2_output = np.random.normal(0, np.sqrt(2.0 / self.hidden_size2), 
                                                       (self.hidden_size2, self.output_size))
        
        # Bias inicializados con sesgo hacia "RUN" (acción más común)
        self.bias_hidden1 = np.zeros((1, self.hidden_size1))
        self.bias_hidden2 = np.zeros((1, self.hidden_size2))
        
        # Bias de salida con sesgo hacia RUN (índice 2)
        self.bias_output = np.array([[-0.5, -0.5, 0.3]])  # Favorece RUN inicialmente

        # ============================================================================================

    def think(self, distance_x, distance_y, game_speed, dino_y, obstacle_type=0, dino_state="RUN"):
        # ======================== PROCESS INFORMATION SENSED TO ACT =============================
        
        # Lógica especial para pájaros - mantener agachado hasta que pase
        if obstacle_type == 2:  # Pájaro
            # Si el pájaro está cerca (menos de 200 px) y el dino no está saltando
            if distance_x < 200 and distance_x > -50 and dino_y > 300:  # dino_y > 300 means on ground
                return "DUCK"  # Mantener agachado
            # Si ya está agachado y el pájaro aún está cerca, seguir agachado
            elif dino_state == "DUCK" and distance_x > -100 and distance_x < 250:
                return "DUCK"
        
        # Normalizar las entradas para mejorar el entrenamiento
        # Distancia X normalizada (0-600 píxeles para dar más urgencia)
        norm_distance_x = max(0, min(distance_x / 600.0, 1.0))
        
        # Distancia Y normalizada (-200 a 200 píxeles aproximadamente)
        norm_distance_y = (distance_y + 200) / 400.0
        
        # Velocidad del juego normalizada (típicamente entre 15-35)
        norm_game_speed = max(0, min(game_speed / 35.0, 1.0))
        
        # Altura del dinosaurio normalizada (suelo = 310, salto máximo ≈ 210)
        norm_dino_y = max(0, min((350 - dino_y) / 140.0, 1.0))
        
        # Tipo de obstáculo normalizado (0=cactus pequeño, 1=cactus grande, 2=pájaro)
        norm_obstacle_type = obstacle_type / 2.0
        
        # Crear vector de entrada
        inputs = np.array([[norm_distance_x, norm_distance_y, norm_game_speed, 
                           norm_dino_y, norm_obstacle_type]])
        
        # Forward propagation con arquitectura más profunda
        # Primera capa oculta
        hidden1_input = np.dot(inputs, self.weights_input_hidden1) + self.bias_hidden1
        hidden1_output = self.relu(hidden1_input)
        
        # Segunda capa oculta
        hidden2_input = np.dot(hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        hidden2_output = self.relu(hidden2_input)
        
        # Capa de salida
        output_layer_input = np.dot(hidden2_output, self.weights_hidden2_output) + self.bias_output
        result = self.softmax(output_layer_input)
        
        # ========================================================================================
        return self.act(result, obstacle_type, distance_x)

    def relu(self, x):
        """Función de activación ReLU (mejor para capas ocultas)"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Función de activación sigmoid"""
        # Clip para evitar overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        """Función de activación softmax para la capa de salida"""
        # Restar el máximo para estabilidad numérica
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def act(self, output, obstacle_type=0, distance_x=1000):
        # ======================== USE THE ACTIVATION FUNCTION TO ACT =============================
        
        # La salida es un vector de probabilidades [prob_jump, prob_duck, prob_run]
        probabilities = output[0]
        
        exploration_rate = 0.01 if distance_x > 500 else 0.02
        
        # Lógica especial para diferentes tipos de obstáculos
        if obstacle_type == 2 and distance_x < 300:  # Pájaro cercano
            # Para pájaros, fuertemente favorecer DUCK
            if distance_x < 150:  # Muy cerca
                return "DUCK"
            elif probabilities[1] > 0.2:  # Si hay algo de confianza en DUCK
                return "DUCK"
        
        # Mejorar lógica de decisión basada en las probabilidades
        exploration_rate = 0.03  # 3% de exploración reducida
        
        if np.random.random() < exploration_rate:
            # Exploración ocasional
            action = np.random.choice(3, p=probabilities)
        else:
            # Seleccionar la mejor acción
            action = np.argmax(probabilities)
            
            # Lógica de seguridad por tipo de obstáculo
            max_prob = np.max(probabilities)
            if max_prob < 0.4:  # Si está indeciso
                if obstacle_type == 0 or obstacle_type == 1:  # Cactus
                    if distance_x < 100:  # Muy cerca
                        action = 0  # JUMP
                    else:
                        action = 2  # RUN
                elif obstacle_type == 2:  # Pájaro
                    if distance_x < 200:
                        action = 1  # DUCK
                    else:
                        action = 2  # RUN
                else:
                    action = 2  # RUN por defecto
        
        # =========================================================================================
        if (action == 0):
            return "JUMP"
        elif (action == 1):
            return "DUCK"
        elif (action == 2):
            return "RUN"