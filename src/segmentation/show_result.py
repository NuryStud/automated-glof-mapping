import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

roi_id = 11
# Carpetas de entrada
image_folder = f"images_rgb_water/{roi_id}/"
mask_folder = f"sam_results/images_rgb_water/{roi_id}/"

# image_files = ["2016-05-29.jpg","2024-05-22"]

# Obtener lista ordenada de archivos
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

# Inicializar el índiceS
current_idx = 0

# Función para mostrar imagen + máscara
def show_image_with_mask(idx):
    img_name = image_files[idx]
    base_name = os.path.splitext(img_name)[0]
    img_path = os.path.join(image_folder, img_name)
    mask_path = os.path.join(mask_folder, base_name + ".npz")

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not os.path.exists(mask_path):
        print(f"No se encontró máscara para {img_name}")
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        mask = np.load(mask_path)["mask"]

    plt.clf()
    plt.title(f"{img_name} - Máscara con IDs únicos")

    # Mostrar imagen original
    plt.imshow(image)

    # Crear máscara colorida
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    unique_ids = np.unique(mask)
    for i in unique_ids:
        if i == 0:
            continue
        color = np.random.randint(0, 255, size=3)
        colored_mask[mask == i] = color

    # Superponer con transparencia
    plt.imshow(colored_mask, alpha=0.5)
    plt.axis("off")
    plt.draw()

# Función de eventos con teclado
def on_key(event):
    global current_idx
    if event.key == 'right':
        current_idx = (current_idx + 1) % len(image_files)
    elif event.key == 'left':
        current_idx = (current_idx - 1) % len(image_files)
    show_image_with_mask(current_idx)

# Mostrar primera imagen
fig = plt.figure(figsize=(10, 8))
show_image_with_mask(current_idx)

# Conectar evento del teclado
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
# plt.savefig(f"sam_output_0_{current_idx}.png")