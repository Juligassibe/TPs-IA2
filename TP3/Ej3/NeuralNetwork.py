import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.initialize()

    def initialize(self):
        # ======================== ARQUITECTURA DE LA RED =============================
        self.input_size = 5       # distance_x, distance_y, game_speed, dino_y, relative_speed
        self.hidden_size1 = 24    # Capa oculta 1 (mayor capacidad de aprendizaje)
        self.hidden_size2 = 16    # Capa oculta 2 (compresión de características)
        self.output_size = 3      # JUMP, DUCK, RUN
        
        # ===================== INICIALIZACIÓN DE PESOS (Xavier/Glorot) ===============
        # Pesos entre capa de entrada y primera capa oculta
        self.weights_input_hidden1 = np.random.normal(
            0, 
            np.sqrt(2.0 / (self.input_size + self.hidden_size1)),
            (self.input_size, self.hidden_size1)
        )
        
        # Pesos entre primera y segunda capa oculta
        self.weights_hidden1_hidden2 = np.random.normal(
            0, 
            np.sqrt(2.0 / (self.hidden_size1 + self.hidden_size2)),
            (self.hidden_size1, self.hidden_size2)
        )
        
        # Pesos entre segunda capa oculta y capa de salida
        self.weights_hidden2_output = np.random.normal(
            0, 
            np.sqrt(2.0 / (self.hidden_size2 + self.output_size)),
            (self.hidden_size2, self.output_size)
        )
        
        # ======================= INICIALIZACIÓN DE SESGOS ===========================
        self.bias_hidden1 = np.zeros((1, self.hidden_size1))
        self.bias_hidden2 = np.zeros((1, self.hidden_size2))
        
        # Sesgo estratégico en capa de salida (favorece RUN por defecto)
        self.bias_output = np.array([[-0.1, -0.1, 0.2]])  # [JUMP, DUCK, RUN]
        
        # =================== NEURONAS ESPECIALIZADAS EN ALTURA ======================
        height_sensitive_neurons = np.random.choice(
            self.hidden_size1, 
            self.hidden_size1 // 3, 
            replace=False
        )
        
        for neuron_idx in height_sensitive_neurons:
            # Entrada de altura (distance_y) con peso negativo fuerte
            self.weights_input_hidden1[1, neuron_idx] = np.random.normal(-2.0, 0.5)
    
    # ========================= FUNCIONES DE ACTIVACIÓN ==============================
    def relu(self, x):
        """Función de activación ReLU (Rectified Linear Unit)"""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Función de activación Softmax (normalización a probabilidades)"""
        # Estabilización numérica
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    
    # ======================== PROPAGACIÓN HACIA ADELANTE ===========================
    def forward(self, inputs):
        """Propagación hacia adelante a través de la red"""
        # Capa oculta 1
        hidden1 = np.dot(inputs, self.weights_input_hidden1) + self.bias_hidden1
        hidden1_act = self.relu(hidden1)
        
        # Capa oculta 2
        hidden2 = np.dot(hidden1_act, self.weights_hidden1_hidden2) + self.bias_hidden2
        hidden2_act = self.relu(hidden2)
        
        # Capa de salida
        output = np.dot(hidden2_act, self.weights_hidden2_output) + self.bias_output
        return self.softmax(output)
    
    # ========================== TOMA DE DECISIONES ================================
    def think(self, distance_x, distance_y, game_speed, dino_y, dino_state="RUN"):
        """
        Procesa las entradas del entorno y decide una acción
        
        Args:
            distance_x: Distancia horizontal al obstáculo más cercano
            distance_y: Diferencia vertical con el obstáculo
            game_speed: Velocidad actual del juego
            dino_y: Posición vertical actual del dinosaurio
            dino_state: Estado actual del dinosaurio ("RUN", "JUMP", "DUCK")
        """
        # Normalización de entradas
        norm_distance_x = np.clip(distance_x / 600.0, 0, 1)
        norm_distance_y = np.clip((distance_y + 150) / 300.0, 0, 1)
        norm_game_speed = np.clip(game_speed / 35.0, 0, 1)
        norm_dino_y = np.clip((350 - dino_y) / 140.0, 0, 1)
        
        # Velocidad relativa (nueva entrada importante)
        relative_speed = np.clip(game_speed / max(distance_x, 1), 0, 1)
        
        # Vector de entrada
        inputs = np.array([[norm_distance_x, norm_distance_y, norm_game_speed, 
                           norm_dino_y, relative_speed]])
        
        # Propagación hacia adelante
        output_probs = self.forward(inputs)
        
        # Selección de acción
        return self.act(output_probs, distance_x, dino_state)
        
    def act(self, output_probs, distance_x=1000, dino_state="RUN"):
        """
        Selecciona una acción basada en las probabilidades de salida
        
        Args:
            output_probs: Probabilidades de salida [JUMP, DUCK, RUN]
            distance_x: Distancia al obstáculo (para emergencias)
            dino_state: Estado actual del dinosaurio
        """
        probabilities = output_probs[0]
        exploration_rate = 0.02  # 2% de probabilidad de exploración
        
        # Exploración controlada
        if np.random.random() < exploration_rate:
            action_idx = np.random.choice(3, p=probabilities)
        else:
            # Explotación: acción con mayor probabilidad
            action_idx = np.argmax(probabilities)
            max_prob = probabilities[action_idx]
            
            # Estrategia de emergencia (obstáculo muy cerca)
            if max_prob < 0.45 and distance_x < 80:
                # Si ya está saltando, agacharse puede ser más seguro
                if dino_state == "JUMP" and probabilities[1] > 0.1:
                    action_idx = 1  # DUCK
                else:
                    # Elegir entre saltar o agacharse
                    action_idx = 0 if probabilities[0] > probabilities[1] else 1
        
        # Mapear índice a acción
        actions = ["JUMP", "DUCK", "RUN"]
        return actions[action_idx]