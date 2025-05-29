import random
import numpy as np
from SaveLoad import SaveLoadManager

# Inicializar sistema de guardado/carga
save_manager = SaveLoadManager()
avg_scores_history = []

def updateNetwork(population, current_generation=None):
    # ===================== ESTA FUNCIÓN RECIBE UNA POBLACIÓN A LA QUE SE DEBEN APLICAR MECANISMOS DE SELECCIÓN, =================
    # ===================== CRUCE Y MUTACIÓN. LA ACTUALIZACIÓN DE LA POBLACIÓN SE APLICA EN LA MISMA VARIABLE ====================
    
    if current_generation and current_generation > 10:
        elite_size = max(3, len(population) // 6)  # Más élite después de 10 generaciones
    else:
        elite_size = max(2, len(population) // 8)  # Menos élite al principio

    mutation_rate = 0.20 if current_generation and current_generation < 15 else 0.12

    global avg_scores_history
    
    # Ordenar población por fitness (score)
    population.sort(key=lambda x: x.score, reverse=True)
    
    # Calcular estadísticas de la generación
    best_score = population[0].score if population else 0
    avg_score = sum(dino.score for dino in population) / len(population) if population else 0
    avg_scores_history.append(avg_score)
    
    # Mantener solo las últimas 50 generaciones en historial
    if len(avg_scores_history) > 50:
        avg_scores_history = avg_scores_history[-50:]
    
    print(f"💪 Gen completada - Mejor: {best_score}, Promedio: {avg_score:.1f}")
    
    # GUARDAR PROGRESO AUTOMÁTICAMENTE
    # Usar generation pasado como parámetro para evitar import circular
    generation_to_save = current_generation if current_generation else 1
    save_manager.save_progress(population, generation_to_save, best_score, avg_scores_history)
    
    # Seleccionar los mejores individuos (elitismo)
    elite_size = max(2, len(population) // 8)  # Top 12.5%
    elite = population[:elite_size]
    
    # Seleccionar padres para reproducción (40% mejores)
    parents = select_fittest(population)
    
    # Crear nueva generación
    new_population = []
    
    # Mantener élite sin cambios
    for dino in elite:
        new_population.append(dino)
        # Resetear para nueva generación pero mantener pesos
        dino.score = 0
    
    # Llenar el resto con cruce y mutación
    while len(new_population) < len(population):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = evolve(parent1, parent2)
        new_population.append(child)
    
    # Reemplazar población original
    for i in range(len(population)):
        if i < len(new_population):
            # Copiar propiedades del dinosaurio pero mantener referencias
            copy_neural_network(new_population[i], population[i])
            population[i].score = 0
            population[i].alive = True

    # =============================================================================================================================

def load_previous_progress(population):
    """Carga progreso previo si existe"""
    global avg_scores_history
    
    save_data = save_manager.load_progress()
    if save_data:
        # Restaurar población élite
        success = save_manager.restore_population(population, save_data)
        
        if success:
            # Restaurar historial de promedios
            avg_scores_history = save_data.get('avg_scores_history', [])
            
            # Mostrar resumen
            print(save_manager.get_stats_summary(save_data))
            print("=" * 50)
            
            return save_data['generation'], save_data['best_score']
    
    print("🚀 Iniciando entrenamiento desde cero...")
    return 1, 0

def select_fittest(population):
    # ===================== FUNCIÓN DE SELECCIÓN =====================
    
    # Selección por torneo + selección proporcional
    population_size = len(population)
    
    # Tomar el 50% superior basado en score
    sorted_population = sorted(population, key=lambda x: x.score, reverse=True)
    selection_size = max(2, population_size // 2)
    
    # Selección con probabilidades proporcionales al fitness
    selected = []
    scores = [dino.score + 1 for dino in sorted_population[:selection_size]]  # +1 para evitar score 0
    total_score = sum(scores)
    
    if total_score > 0:
        probabilities = [score / total_score for score in scores]
        
        # Seleccionar individuos basado en probabilidades
        for i in range(selection_size):
            # Selección por ruleta
            r = random.random()
            cumulative_prob = 0
            for j, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(sorted_population[j])
                    break
    else:
        # Si todos tienen score 0, seleccionar aleatoriamente
        selected = random.sample(sorted_population[:selection_size], 
                               min(selection_size, len(sorted_population)))
    
    return selected if selected else sorted_population[:selection_size]

    # ================================================================

def evolve(element1, element2):
    # ===================== FUNCIÓN DE CRUCE Y MUTACIÓN =====================
    
    # Crear nuevo dinosaurio hijo
    child = type(element1)(element1.id)  # Crear nueva instancia
    
    # CRUCE (Crossover) - Combinar pesos de los padres
    # Cruce uniforme para cada matriz de pesos
    
    # Pesos input -> hidden1
    mask1 = np.random.random(element1.weights_input_hidden1.shape) < 0.5
    child.weights_input_hidden1 = np.where(mask1, 
                                         element1.weights_input_hidden1, 
                                         element2.weights_input_hidden1)
    
    # Pesos hidden1 -> hidden2
    mask2 = np.random.random(element1.weights_hidden1_hidden2.shape) < 0.5
    child.weights_hidden1_hidden2 = np.where(mask2, 
                                           element1.weights_hidden1_hidden2, 
                                           element2.weights_hidden1_hidden2)
    
    # Pesos hidden2 -> output
    mask3 = np.random.random(element1.weights_hidden2_output.shape) < 0.5
    child.weights_hidden2_output = np.where(mask3, 
                                          element1.weights_hidden2_output, 
                                          element2.weights_hidden2_output)
    
    # Bias
    mask_b1 = np.random.random(element1.bias_hidden1.shape) < 0.5
    child.bias_hidden1 = np.where(mask_b1, element1.bias_hidden1, element2.bias_hidden1)
    
    mask_b2 = np.random.random(element1.bias_hidden2.shape) < 0.5
    child.bias_hidden2 = np.where(mask_b2, element1.bias_hidden2, element2.bias_hidden2)
    
    mask_b3 = np.random.random(element1.bias_output.shape) < 0.5
    child.bias_output = np.where(mask_b3, element1.bias_output, element2.bias_output)
    
    # MUTACIÓN - Agregar ruido aleatorio
    mutation_rate = 0.15  # 15% de probabilidad de mutación
    mutation_strength = 0.3  # Intensidad de la mutación
    
    # Mutar cada peso con cierta probabilidad
    if np.random.random() < mutation_rate:
        mutation_mask = np.random.random(child.weights_input_hidden1.shape) < 0.1
        child.weights_input_hidden1 += mutation_mask * np.random.normal(0, mutation_strength, 
                                                                       child.weights_input_hidden1.shape)
    
    if np.random.random() < mutation_rate:
        mutation_mask = np.random.random(child.weights_hidden1_hidden2.shape) < 0.1
        child.weights_hidden1_hidden2 += mutation_mask * np.random.normal(0, mutation_strength, 
                                                                         child.weights_hidden1_hidden2.shape)
    
    if np.random.random() < mutation_rate:
        mutation_mask = np.random.random(child.weights_hidden2_output.shape) < 0.1
        child.weights_hidden2_output += mutation_mask * np.random.normal(0, mutation_strength, 
                                                                        child.weights_hidden2_output.shape)
    
    # Mutar bias
    if np.random.random() < mutation_rate:
        child.bias_hidden1 += np.random.normal(0, mutation_strength * 0.1, child.bias_hidden1.shape)
    
    if np.random.random() < mutation_rate:
        child.bias_hidden2 += np.random.normal(0, mutation_strength * 0.1, child.bias_hidden2.shape)
    
    if np.random.random() < mutation_rate:
        child.bias_output += np.random.normal(0, mutation_strength * 0.1, child.bias_output.shape)
    
    return child

    # ===============================================================

def copy_neural_network(source, target):
    """Copia los pesos de la red neuronal de source a target"""
    target.weights_input_hidden1 = source.weights_input_hidden1.copy()
    target.weights_hidden1_hidden2 = source.weights_hidden1_hidden2.copy()
    target.weights_hidden2_output = source.weights_hidden2_output.copy()
    target.bias_hidden1 = source.bias_hidden1.copy()
    target.bias_hidden2 = source.bias_hidden2.copy()
    target.bias_output = source.bias_output.copy()