import random
import numpy as np
from SaveLoad import SaveLoadManager

# Inicializar sistema de guardado/carga
save_manager = SaveLoadManager()
avg_scores_history = []

def calculate_smart_fitness(dino, final_score, action_history=[]):
    base_fitness = final_score
    survival_bonus = final_score * 0.1
    jump_penalty = 0
    
    if action_history:
        total_actions = len(action_history)
        jump_count = action_history.count("JUMP")
        
        if jump_count > total_actions * 0.4:
            jump_penalty = final_score * 0.05
        
        strategy_variety = len(set(action_history)) / 3.0
        variety_bonus = final_score * strategy_variety * 0.02
    else:
        variety_bonus = 0
    
    smart_fitness = base_fitness + survival_bonus - jump_penalty + variety_bonus
    return max(0, smart_fitness)

def updateNetwork(population, current_generation=None):
    # ===================== FUNCIÓN PRINCIPAL DE ALGORITMO GENÉTICO HONESTO =================
    
    global avg_scores_history
    
    # Calcular fitness inteligente para cada dinosaurio
    for dino in population:
        dino.smart_fitness = calculate_smart_fitness(dino, dino.score)
    
    # Ordenar por fitness inteligente
    population.sort(key=lambda x: x.smart_fitness, reverse=True)
    
    # Calcular estadísticas
    best_score = population[0].score if population else 0
    best_fitness = population[0].smart_fitness if population else 0
    avg_score = sum(dino.score for dino in population) / len(population) if population else 0
    avg_fitness = sum(dino.smart_fitness for dino in population) / len(population) if population else 0
    
    avg_scores_history.append(avg_score)
    
    # Mantener solo las últimas 50 generaciones en historial
    if len(avg_scores_history) > 50:
        avg_scores_history = avg_scores_history[-50:]
    
    print(f"💪 Gen completada - Mejor: {best_score} (fitness: {best_fitness:.1f}), Promedio: {avg_score:.1f}")
    
    # Análisis de progreso cada 5 generaciones
    if current_generation and current_generation % 5 == 0:
        print(f"📊 Análisis Gen {current_generation}:")
        print(f"   🏆 Fitness promedio: {avg_fitness:.1f}")
        if len(avg_scores_history) >= 5:
            recent_trend = avg_scores_history[-1] - avg_scores_history[-5]
            trend_emoji = "📈" if recent_trend > 0 else "📉" if recent_trend < 0 else "➡️"
            print(f"   {trend_emoji} Tendencia (5 gen): {recent_trend:+.1f}")
    
    # GUARDAR PROGRESO AUTOMÁTICAMENTE
    generation_to_save = current_generation if current_generation else 1
    save_manager.save_progress(population, generation_to_save, best_score, avg_scores_history)
    
    # Parámetros adaptativos honestos
    if current_generation and current_generation > 10:
        elite_size = max(3, len(population) // 6)  # Más élite después de 10 generaciones (16.7%)
    else:
        elite_size = max(2, len(population) // 8)  # Menos élite al principio (12.5%)
    
    # Seleccionar los mejores individuos (elitismo)
    elite = population[:elite_size]
    
    # Seleccionar padres para reproducción (basado en fitness inteligente)
    parents = select_fittest(population)
    
    # Crear nueva generación
    new_population = []
    
    # Mantener élite sin cambios
    for dino in elite:
        new_population.append(dino)
        dino.score = 0
        dino.smart_fitness = 0
        if hasattr(dino, 'last_actions'):
            dino.last_actions = []  # Reset action history
    
    # Llenar el resto con cruce y mutación
    while len(new_population) < len(population):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = evolve(parent1, parent2, current_generation)
        new_population.append(child)
    
    # Reemplazar población original
    for i in range(len(population)):
        if i < len(new_population):
            copy_neural_network(new_population[i], population[i])
            population[i].score = 0
            population[i].smart_fitness = 0
            population[i].alive = True
            if not hasattr(population[i], 'last_actions'):
                population[i].last_actions = []

    # =============================================================================================================================

def select_fittest(population):
    """Selección honesta de padres basada en fitness inteligente"""
    population_size = len(population)
    
    # Ordenar por fitness inteligente
    sorted_population = sorted(population, key=lambda x: x.smart_fitness, reverse=True)
    selection_size = max(2, population_size // 2)  # Top 50%
    
    # Selección proporcional honesta
    selected = []
    scores = [dino.smart_fitness + 1 for dino in sorted_population[:selection_size]]  # +1 para evitar 0
    total_score = sum(scores)
    
    if total_score > 0:
        probabilities = [score / total_score for score in scores]
        
        # Selección por ruleta ponderada
        for i in range(selection_size):
            r = random.random()
            cumulative_prob = 0
            for j, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(sorted_population[j])
                    break
    else:
        # Fallback: selección aleatoria del top 50%
        selected = random.sample(sorted_population[:selection_size], 
                               min(selection_size, len(sorted_population)))
    
    return selected if selected else sorted_population[:selection_size]

def evolve(element1, element2, current_generation=None):
    """Función de cruce y mutación honesta"""
    
    # Crear nuevo dinosaurio hijo
    child = type(element1)(element1.id)
    
    # CRUCE uniforme (honesto)
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
    
    # Cruce de bias
    mask_b1 = np.random.random(element1.bias_hidden1.shape) < 0.5
    child.bias_hidden1 = np.where(mask_b1, element1.bias_hidden1, element2.bias_hidden1)
    
    mask_b2 = np.random.random(element1.bias_hidden2.shape) < 0.5
    child.bias_hidden2 = np.where(mask_b2, element1.bias_hidden2, element2.bias_hidden2)
    
    mask_b3 = np.random.random(element1.bias_output.shape) < 0.5
    child.bias_output = np.where(mask_b3, element1.bias_output, element2.bias_output)
    
    # MUTACIÓN adaptativa honesta
    mutation_rate = 0.18 if current_generation and current_generation < 15 else 0.12
    mutation_strength = 0.25  # Intensidad moderada
    
    # Aplicar mutación a pesos
    if np.random.random() < mutation_rate:
        mutation_mask = np.random.random(child.weights_input_hidden1.shape) < 0.08
        child.weights_input_hidden1 += mutation_mask * np.random.normal(0, mutation_strength, 
                                                                       child.weights_input_hidden1.shape)
    
    if np.random.random() < mutation_rate:
        mutation_mask = np.random.random(child.weights_hidden1_hidden2.shape) < 0.08
        child.weights_hidden1_hidden2 += mutation_mask * np.random.normal(0, mutation_strength, 
                                                                         child.weights_hidden1_hidden2.shape)
    
    if np.random.random() < mutation_rate:
        mutation_mask = np.random.random(child.weights_hidden2_output.shape) < 0.08
        child.weights_hidden2_output += mutation_mask * np.random.normal(0, mutation_strength, 
                                                                        child.weights_hidden2_output.shape)
    
    # Mutar bias con menor intensidad
    if np.random.random() < mutation_rate:
        child.bias_hidden1 += np.random.normal(0, mutation_strength * 0.1, child.bias_hidden1.shape)
    
    if np.random.random() < mutation_rate:
        child.bias_hidden2 += np.random.normal(0, mutation_strength * 0.1, child.bias_hidden2.shape)
    
    if np.random.random() < mutation_rate:
        child.bias_output += np.random.normal(0, mutation_strength * 0.1, child.bias_output.shape)
    
    # Inicializar tracking de acciones para fitness inteligente
    child.last_actions = []
    child.smart_fitness = 0
    
    return child

def copy_neural_network(source, target):
    """Copia los pesos de la red neuronal honestamente"""
    target.weights_input_hidden1 = source.weights_input_hidden1.copy()
    target.weights_hidden1_hidden2 = source.weights_hidden1_hidden2.copy()
    target.weights_hidden2_output = source.weights_hidden2_output.copy()
    target.bias_hidden1 = source.bias_hidden1.copy()
    target.bias_hidden2 = source.bias_hidden2.copy()
    target.bias_output = source.bias_output.copy()

def load_previous_progress(population):
    """Carga progreso previo honestamente"""
    global avg_scores_history
    
    save_data = save_manager.load_progress()
    if save_data:
        success = save_manager.restore_population(population, save_data)
        
        if success:
            # Inicializar tracking en población restaurada
            for dino in population:
                if not hasattr(dino, 'last_actions'):
                    dino.last_actions = []
                if not hasattr(dino, 'smart_fitness'):
                    dino.smart_fitness = 0
            
            avg_scores_history = save_data.get('avg_scores_history', [])
            print(save_manager.get_stats_summary(save_data))
            print("🧬 Sistema de fitness inteligente activado")
            print("=" * 50)
            
            return save_data['generation'], save_data['best_score']
    
    print("🚀 Iniciando entrenamiento honesto desde cero...")
    print("🧬 Fitness inteligente: Penaliza ineficiencia SIN hacer trampa")
    
    # Inicializar tracking en nueva población
    for dino in population:
        dino.last_actions = []
        dino.smart_fitness = 0
    
    return 1, 0