import subprocess
try:
    import tensorflow as tf
except ImportError as err:
    subprocess.check_call(['pip', 'install', 'tensorflow'])
    subprocess.check_call(['pip', 'install', 'Pillow'])
    import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import random

import matplotlib.pyplot as plt

# Rutas de las carpetas
source_dir = "images"

train_dir = source_dir + "/train/"
test_dir = source_dir + "/test/"

# Clases (nombres de las subcarpetas)
classes = ["up", "down", "right"]

# Crea los directorios de entrenamiento y prueba si no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Crea los directorios de entrenamiento y prueba si no existen
os.makedirs(train_dir + classes[0], exist_ok=True)
os.makedirs(train_dir + classes[1], exist_ok=True)
os.makedirs(train_dir + classes[2], exist_ok=True)
os.makedirs(test_dir + classes[0], exist_ok=True)
os.makedirs(test_dir + classes[1], exist_ok=True)
os.makedirs(test_dir + classes[2], exist_ok=True)

# Proporción de imágenes para entrenamiento y prueba
train_ratio = 0.8

# Parámetros para el modelo
batch_size = 32
image_size = (224, 224)
input_shape = image_size + (1,)  # Tamaño de la imagen con un solo canal para escala de grises

# Función para cargar imágenes y convertirlas a escala de grises
def load_and_preprocess_image(file_path, target_size):
    img = load_img(file_path, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img)
    return img_array / 255.0  # Normaliza los valores de píxeles entre 0 y 1

# Iterar sobre las subcarpetas
for class_name in classes:
    # Ruta de la subcarpeta de origen
    source_class_dir = os.path.join(source_dir, class_name)
    
    # Obtener la lista de imágenes en la subcarpeta de origen
    images = os.listdir(source_class_dir)
    
    # Mezclar aleatoriamente las imágenes
    random.shuffle(images)
    
    # Calcular el número de imágenes para entrenamiento
    num_train_images = int(len(images) * train_ratio)
    
    # Iterar sobre las imágenes para entrenamiento
    for img_name in images[:num_train_images]:
        # Ruta de la imagen de origen
        src_img_path = os.path.join(source_class_dir, img_name)
        # Ruta de destino para la imagen de entrenamiento
        dest_train_path = os.path.join(train_dir + class_name, f"{img_name}")
        # Mover la imagen a la carpeta de entrenamiento y renombrarla
        img_array = load_and_preprocess_image(src_img_path, image_size)
        tf.keras.preprocessing.image.save_img(dest_train_path, img_array)
    
    # Iterar sobre las imágenes para prueba
    for img_name in images[num_train_images:]:
        # Ruta de la imagen de origen
        src_img_path = os.path.join(source_class_dir, img_name)
        # Ruta de destino para la imagen de prueba
        dest_test_path = os.path.join(test_dir + class_name, f"{img_name}")
        # Mover la imagen a la carpeta de prueba y renombrarla
        img_array = load_and_preprocess_image(src_img_path, image_size)
        # tf.keras.preprocessing.image.save_img(dest_train_path, img_array)
        tf.keras.preprocessing.image.save_img(dest_test_path, img_array)

# Crear generadores de datos
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True)  # Se especifica el modo de color escala de grises

print("Mapa de clases:", train_generator.class_indices)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False)  # Se especifica el modo de color escala de grises

print("Validation generator samples:", validation_generator.samples)

# ========================== Construir el modelo ==========================================
model = tf.keras.models.Sequential([
    #capa convulucionales y pooling
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), #Cant. de nuecleos, tamaño del núcleo, función de activación, forma de entrada
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
        # Cuarto bloque conv + pooling
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(), # Convierte el resultado de una imagen cuadrada a un vector simple (unidimesional) que puede usarse con las capas regulares de adelante.
    tf.keras.layers.Dense(128, activation='relu'), #cant. de neuronas, funcion de activación
    tf.keras.layers.Dropout(0.5),  # En cada epocs o ciclo de entrenamiento, se desactivan aleatoriamente el 50% de las neuronas para evitar el sobreajuste (overfitting)
                                   # Esto se usa ya que despues del entrenamiento, algunos pesos son muy grandes y otros muy pequeños, lo que puede causar que el modelo no generalice bien.
                                   # Por lo que esos pesos pequeños no tiene mucha influencia en las decisiones de la red.
                                   # Entonces este apagado aleatorio permite que esas neutronas trabajan más y asi hacemos que el conjunto de red sea más resiliente. 
                                   
        # Segunda capa densa para mejor capacidad de representación
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(len(classes), activation='softmax')  # Capa de salida con softmax para clasificación multiclase
])
# ==========================================================================================

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks para early stopping (parada temprana) y guardar mejor modelo
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

# Entrenar el modelo
history = model.fit(train_generator, epochs=5, validation_data=validation_generator, callbacks=callbacks)

# Guardar el modelo
model.save('tensorflow_nn.h5')

def plot_training_curves(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Graficar curvas
plot_training_curves(history)