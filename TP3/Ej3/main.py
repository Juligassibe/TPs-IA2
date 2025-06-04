import subprocess
try:
    import pygame
except ImportError as err:
    subprocess.check_call(['pip', 'install', 'pygame'])
    import pygame

import os
import random
import numpy as np
from Dinosaur import Dinosaur
from Cloud import Cloud
from Bird import Bird
from SmallCactus import SmallCactus
from LargeCactus import LargeCactus
from Genetic import updateNetwork, load_previous_progress
from ImageCapture import ImageCapture

screen_spawn_position = (100, 100)
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % screen_spawn_position
pygame.init()

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("DinoGame")
generation = 1
bestScore = 0
playMode = "X"

imageCapture = ImageCapture(screen_spawn_position)

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

def populate(population_size):
    """Crea población con diversidad inicial mejorada"""
    population = []
    for i in range(population_size):
        while True:
            R = random.randint(0, 255)
            G = random.randint(0, 255)
            B = random.randint(0, 255)
            brightness = 0.299 * R + 0.587 * G + 0.114 * B
            if brightness < 180:  # Evitar colores demasiado claros
                break
        color = (R, G, B)
        
        dino = Dinosaur(i, color, True)
        
        # MEJORA SILENCIOSA: Agregar diversidad inicial en los pesos (excepto el primero)
        if i > 0:
            noise_factor = 0.08  # Diversidad sutil
            dino.weights_input_hidden1 += np.random.normal(0, noise_factor, 
                                                          dino.weights_input_hidden1.shape)
            dino.weights_hidden1_hidden2 += np.random.normal(0, noise_factor, 
                                                           dino.weights_hidden1_hidden2.shape)
            dino.weights_hidden2_output += np.random.normal(0, noise_factor, 
                                                          dino.weights_hidden2_output.shape)
            
            # 🚀 DIVERSIDAD: Algunos dinos con sesgo hacia agacharse
            if i % 5 == 0:  # Cada 15vo dinosaurio
                dino.weights_hidden2_output[:, 1] += 0.45  # Más probabilidad de agacharse
        
        population.append(dino)
    
    return population

# ======================== MEJORA: POBLACIÓN AUMENTADA ======================
population_number = 150  
# ========================================================================
population = populate(population_number)
player = Dinosaur(0)
callUpdateNetwork = False

# CARGAR PROGRESO PREVIO AUTOMÁTICAMENTE (solo para modo genético)
def load_genetic_progress():
    global generation, bestScore
    if playMode != 'm' and playMode != 'c' and playMode != 'a':  # Solo para modo genético
        try:
            loaded_generation, loaded_best_score = load_previous_progress(population)
            if loaded_generation > 1:
                generation = loaded_generation
                bestScore = loaded_best_score
                print(f"🔄 Continuando desde generación {generation} con mejor score: {bestScore}")
        except Exception as e:
            print(f"⚠️  Iniciando desde cero: {e}")

def calculate_enhanced_fitness(dino):
    """Fitness que PROTEGE a los dinos experimentales"""
    base_score = dino.score
    
    # Bonificaciones originales
    survival_bonus = getattr(dino, 'frames_survived', 0) * 0.05
    if base_score > 500:
        score_bonus = (base_score - 500) * 1.2
    else:
        score_bonus = 0
    
    # 🛡️ PROTECCIÓN CLAVE: Si intentó agacharse, darle una oportunidad
    duck_protection = 0
    duck_attempts = getattr(dino, 'duck_attempts', 0)
    
    if duck_attempts > 0:
        # Bonus base por experimentar
        duck_protection = 100
        
        # Bonus extra si sobrevivió un tiempo razonable
        if dino.score > 50:
            duck_protection += duck_attempts * 20
        
        # Mega bonus si sobrevivió mucho tiempo agachándose
        if dino.score > 200:
            duck_protection += 200
    
    enhanced = int(base_score + survival_bonus + score_bonus + duck_protection)
    dino.enhanced_score = enhanced
    return enhanced

def gameScreen():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles, population, callUpdateNetwork, generation, bestScore, playMode
    run = True
    clock = pygame.time.Clock()
    game_speed = 20
    cloud = Cloud(SCREEN_WIDTH, game_speed)
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    callUpdateNetwork = True

    # Variables para rastrear el último dinosaurio muerto y el obstáculo que lo mató
    last_dino = None
    killing_obstacle = None

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Puntos: " + str(points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def countSurviving():
        global population
        text = font.render("Vivos: " + str(count_alive(population)), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 65)
        SCREEN.blit(text, textRect)

    def currentGeneration():
        global generation
        text = font.render("Generación: " + str(generation), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 90)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    def deathUpdates(player, obstacle):
        global generation, points, bestScore
        obstacle_params = obstacle.rect
        SCREEN.fill((255, 255, 255))
        background()
        score()
        cloud.draw(SCREEN)
        cloud.update()
        SCREEN.blit(player.image, (player.dino_rect.x, player.dino_rect.y))
        SCREEN.blit(obstacle.image[obstacle.type], (obstacle_params.x, obstacle_params.y))
        pygame.draw.rect(SCREEN, (255, 0, 0), player.dino_rect, 2)
        pygame.draw.rect(SCREEN, (0, 0, 255), obstacle_params, 2)
        pygame.display.update()
        pygame.time.delay(1000)
        generation += 1
        if points > bestScore:
            bestScore = points
        menu()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        SCREEN.fill((255, 255, 255))

        if playMode == 'm' or playMode == 'c':
            userInput = pygame.key.get_pressed()
            player.draw(SCREEN)
            player.update(userInput)
            if playMode == 'c':
                imageCapture.capture(userInput)

        elif playMode == 'a':
            if player.alive:
                player.draw(SCREEN)
                imageCapture.capture_live()
                player.predict()

        else:
            # Modo genético
            for dino in population:
                if not hasattr(dino, 'action_history'):
                    dino.action_history = []
                    
                if dino.alive:
                    dino.draw(SCREEN)
                    
                    # Encontrar el obstáculo más cercano
                    closest_obstacle = None
                    min_distance = float('inf')
                    
                    for obstacle in obstacles:
                        obstacle_params = obstacle.rect
                        dino_params = dino.dino_rect
                        
                        # Calcular distancia horizontal al obstáculo
                        distance_x = obstacle_params.x - dino_params.x
                        
                        # Solo considerar obstáculos que están adelante del dinosaurio
                        if distance_x > -50 and distance_x < min_distance:
                            min_distance = distance_x
                            closest_obstacle = obstacle
                    
                    # Determinar estado actual del dinosaurio
                    current_state = "RUN"
                    if dino.dino_jump:
                        current_state = "JUMP"
                    elif dino.dino_duck:
                        current_state = "DUCK"
                    
                    # Llamar a think() con obstacle_type = 0 siempre
                    if closest_obstacle:
                        obstacle_params = closest_obstacle.rect
                        dino_params = dino.dino_rect
                        distance_x = obstacle_params.x - dino_params.x
                        distance_y = obstacle_params.y - dino_params.y
                        action = dino.think(distance_x, distance_y, game_speed, dino.dino_rect.y, current_state)
                    else:
                        action = dino.think(800, 0, game_speed, dino.dino_rect.y, "RUN")
                    
                    # Registrar acción para análisis posterior
                    if len(dino.action_history) < 1000:
                        dino.action_history.append(action)
                    
                    # Actualizar el dinosaurio
                    dino.update(action)

        # Generar nuevos obstáculos
        if len(obstacles) == 0:
            obstacle_type = random.randint(0, 2)
            if obstacle_type == 0:
                obstacles.append(SmallCactus(SCREEN_WIDTH, game_speed, obstacles))
            elif obstacle_type == 1:
                obstacles.append(LargeCactus(SCREEN_WIDTH, game_speed, obstacles))
            else:
                obstacles.append(Bird(SCREEN_WIDTH, game_speed, obstacles))

        # Variables para rastrear colisiones
        player_killing_obstacle = None
        last_dino = None
        killing_obstacle = None

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            obstacle_params = obstacle.rect

            # Detectar colisiones para el jugador
            if playMode in ['m', 'c', 'a']:
                if player.alive and player.dino_rect.colliderect(obstacle_params):
                    player.alive = False
                    player_killing_obstacle = obstacle
            
            # Detectar colisiones para la población
            else:
                for dino in population:
                    if dino.alive and dino.dino_rect.colliderect(obstacle_params):
                        dino.score = points
                        dino.alive = False
                        last_dino = dino
                        killing_obstacle = obstacle

        # Manejar muerte del jugador
        if playMode in ['m', 'c', 'a'] and not player.alive and player_killing_obstacle:
            deathUpdates(player, player_killing_obstacle)
        
        # Manejar extinción de la población
        elif playMode not in ['m', 'c', 'a'] and count_alive(population) == 0 and last_dino and killing_obstacle:
            deathUpdates(last_dino, killing_obstacle)

        # Actualizar gráficos
        background()
        cloud.draw(SCREEN)
        cloud.update()
        score()

        if playMode not in ['m', 'c', 'a']:
            countSurviving()
            currentGeneration()

        clock.tick(30)
        pygame.display.update()

def menu():
    global callUpdateNetwork, generation, bestScore, playMode, population, population_number
    run = True

    if playMode == 'm' or playMode == 'c' or playMode == 'a':
        player.resetStatus()
    elif playMode not in ['m', 'c', 'a'] and callUpdateNetwork:
        # Actualizar red neuronal con la generación actual
        updateNetwork(population, generation)
        callUpdateNetwork = False
        
        # Resetear dinosaurios para nueva generación
        for dino in population:
            dino.resetStatus()
            dino.frames_survived = 0
            dino.duck_attempts = 0
            dino.action_history = []  # Limpiar historial de acciones
        
    while run:
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 30)

        # Interfaz del menú
        if generation == 1:
            text = font.render("Pulse 'm' para jugar manualmente", True, (0, 0, 0))

            auxText = font.render("'c' para capturar imágenes", True, (0, 0, 0))
            auxTextRect = auxText.get_rect()
            auxTextRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
            SCREEN.blit(auxText, auxTextRect)

            auxText = font.render("'a' para usar el modelo generado por Tensorflow", True, (0, 0, 0))
            auxTextRect = auxText.get_rect()
            auxTextRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100)
            SCREEN.blit(auxText, auxTextRect)

            auxText = font.render("o cualquier otra letra para jugar automáticamente", True, (0, 0, 0))
            auxTextRect = auxText.get_rect()
            auxTextRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 150)
            SCREEN.blit(auxText, auxTextRect)
            
        elif generation > 1:
            text = font.render("Pulse cualquier tecla para reiniciar", True, (0, 0, 0))
            score = font.render("Mejor puntuación: " + str(bestScore), True, (0, 0, 0))
            scoreRect = score.get_rect()
            scoreRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
            SCREEN.blit(score, scoreRect)

        textRect = text.get_rect()
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        SCREEN.blit(text, textRect)
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                if generation == 1:
                    # Primera vez - elegir modo
                    if event.type == pygame.KEYDOWN:
                        playMode = pygame.key.name(event.key)
                    else:  # Click de ratón - modo genético por defecto
                        playMode = "genetic"

                    if playMode in ['m', 'c', 'a']:
                        population = []
                    else:
                        # Cargar progreso genético si existe
                        load_genetic_progress()
                
                gameScreen()

def count_alive(population):
    """Función optimizada para contar dinosaurios vivos"""
    return sum(1 for dino in population if dino.alive)

if __name__ == "__main__":
    menu()