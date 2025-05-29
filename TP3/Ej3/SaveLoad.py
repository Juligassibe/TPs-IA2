import pickle
import os
import numpy as np
from datetime import datetime

class SaveLoadManager:
    def __init__(self, save_file="dinosaur_progress.pkl"):
        self.save_file = save_file
        self.backup_file = "dinosaur_progress_backup.pkl"
        
    def save_progress(self, population, generation, best_score, avg_scores_history=[]):
        """Guarda el progreso completo del entrenamiento"""
        try:
            # Preparar datos para guardar
            save_data = {
                'generation': generation,
                'best_score': best_score,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'avg_scores_history': avg_scores_history,
                'population_data': []
            }
            
            # Guardar solo los mejores individuos (top 20%) para ahorrar espacio
            population_sorted = sorted(population, key=lambda x: x.score, reverse=True)
            elite_count = max(5, len(population) // 5)  # Top 20% o mínimo 5
            
            for i, dino in enumerate(population_sorted[:elite_count]):
                dino_data = {
                    'rank': i + 1,
                    'score': dino.score,
                    'weights_input_hidden1': dino.weights_input_hidden1.copy(),
                    'weights_hidden1_hidden2': dino.weights_hidden1_hidden2.copy(), 
                    'weights_hidden2_output': dino.weights_hidden2_output.copy(),
                    'bias_hidden1': dino.bias_hidden1.copy(),
                    'bias_hidden2': dino.bias_hidden2.copy(),
                    'bias_output': dino.bias_output.copy(),
                    'id': dino.id,
                    'color': getattr(dino, 'color', None)
                }
                save_data['population_data'].append(dino_data)
            
            # Crear backup del archivo anterior
            if os.path.exists(self.save_file):
                try:
                    os.rename(self.save_file, self.backup_file)
                except:
                    pass  # Si no puede hacer backup, continúa
            
            # Guardar nuevo archivo
            with open(self.save_file, 'wb') as f:
                pickle.dump(save_data, f)
                
            print(f" Progreso guardado - Gen {generation}, Mejor: {best_score}")
            return True
            
        except Exception as e:
            print(f" Error al guardar: {e}")
            return False
    
    def load_progress(self):
        """Carga el progreso guardado"""
        try:
            if not os.path.exists(self.save_file):
                print(" No se encontró archivo de progreso previo")
                return None
                
            with open(self.save_file, 'rb') as f:
                save_data = pickle.load(f)
            
            print(f" Progreso cargado:")
            print(f"    Generación: {save_data['generation']}")
            print(f"    Mejor score: {save_data['best_score']}")
            print(f"    Última sesión: {save_data['timestamp']}")
            print(f"    Élite guardada: {len(save_data['population_data'])} individuos")
            
            return save_data
            
        except Exception as e:
            print(f"❌ Error al cargar progreso: {e}")
            # Intentar cargar backup
            try:
                if os.path.exists(self.backup_file):
                    print(" Intentando cargar backup...")
                    with open(self.backup_file, 'rb') as f:
                        save_data = pickle.load(f)
                    print(" Backup cargado exitosamente")
                    return save_data
            except:
                pass
            return None
    
    def restore_population(self, population, save_data):
        """Restaura los mejores individuos en la población actual"""
        try:
            if not save_data or 'population_data' not in save_data:
                return False
                
            elite_data = save_data['population_data']
            restored_count = 0
            
            # Restaurar élite en los primeros individuos de la población
            for i, dino_data in enumerate(elite_data):
                if i < len(population):
                    dino = population[i]
                    
                    # Restaurar pesos y bias
                    dino.weights_input_hidden1 = dino_data['weights_input_hidden1'].copy()
                    dino.weights_hidden1_hidden2 = dino_data['weights_hidden1_hidden2'].copy()
                    dino.weights_hidden2_output = dino_data['weights_hidden2_output'].copy()
                    dino.bias_hidden1 = dino_data['bias_hidden1'].copy()
                    dino.bias_hidden2 = dino_data['bias_hidden2'].copy()
                    dino.bias_output = dino_data['bias_output'].copy()
                    
                    # Restaurar otras propiedades
                    dino.score = 0  # Reset score para nueva generación
                    if dino_data['color'] and hasattr(dino, 'color'):
                        dino.color = dino_data['color']
                    
                    restored_count += 1
            
            print(f" Restaurados {restored_count} dinosaurios élite")
            return True
            
        except Exception as e:
            print(f" Error al restaurar población: {e}")
            return False
    
    def get_stats_summary(self, save_data):
        """Obtiene un resumen de estadísticas"""
        if not save_data:
            return "Sin progreso previo"
            
        summary = f"""
=== RESUMEN DE PROGRESO ===
Generación actual: {save_data['generation']}
Mejor puntuación: {save_data['best_score']}
Última sesión: {save_data['timestamp']}
Élite guardada: {len(save_data.get('population_data', []))} individuos
        """
        
        if 'avg_scores_history' in save_data and save_data['avg_scores_history']:
            recent_avg = save_data['avg_scores_history'][-5:]  # Últimos 5 promedios
            summary += f"📈 Progreso reciente: {[f'{x:.1f}' for x in recent_avg]}\n"
        
        return summary

# Función auxiliar para integrar fácilmente
def setup_save_load():
    """Función helper para configurar el sistema de guardado"""
    return SaveLoadManager()