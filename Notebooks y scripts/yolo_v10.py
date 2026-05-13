import cv2
import torch
from ultralytics import YOLO
from evdev import UInput, ecodes as e

# 1. Configuración de Mapeo (Etiqueta -> Tecla de evdev)
# Puedes añadir todos los que necesites aquí
GESTURE_MAP = {
    "four": e.KEY_LEFT,    # Si detecta celular, mantiene tecla A
    "palm": e.KEY_RIGHT, # Ejemplo para tu futuro modelo de gestos
    "peace": e.KEY_DOWN,
    "one": e.KEY_UP
}

# 2. Configuración del Teclado Virtual
cap_bits = {e.EV_KEY: list(GESTURE_MAP.values())}
try:
    ui = UInput(cap_bits, name='yolo-sustained-input')
    print("✅ Teclado virtual iniciado en modo 'Sostenido'.")
except PermissionError:
    print("❌ Error de permisos. Usa sudo.")
    exit()

# 3. Cargar Modelo
model = YOLO("yolov10n.pt")

# Diccionario para rastrear qué teclas están actualmente presionadas
active_keys = {label: False for label in GESTURE_MAP}

cap_video = cv2.VideoCapture(0)

while True:
    ret, frame = cap_video.read()
    if not ret: break

    # Inferencia (Optimizada para 940MX con imgsz=320 si hay lag)
    results = model(frame, device="cuda", verbose=False, imgsz=416)
    
    # Obtener etiquetas detectadas en este frame específico
    current_frame_labels = set()
    for box in results[0].boxes:
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = model.names[cls]
        
        if conf > 0.5 and label in GESTURE_MAP:
            current_frame_labels.add(label)

    # 4. Lógica de "Mantener Presionado"
    for label, key_code in GESTURE_MAP.items():
        # Si el label está en el frame y no estaba presionado antes -> PRESIONAR
        if label in current_frame_labels and not active_keys[label]:
            ui.write(e.EV_KEY, key_code, 1) # 1 = KEY_DOWN
            ui.syn()
            active_keys[label] = True
            print(f"⬇️  Manteniendo {label} ({key_code})")

        # Si el label DESAPARECIÓ del frame y estaba presionado -> SOLTAR
        elif label not in current_frame_labels and active_keys[label]:
            ui.write(e.EV_KEY, key_code, 0) # 0 = KEY_UP
            ui.syn()
            active_keys[label] = False
            print(f"⬆️  Soltando {label}")

    # Visualización
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Sustained Input", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27: break

# Al cerrar, soltamos todas las teclas por seguridad (evitar que la PC quede loca)
for label, key_code in GESTURE_MAP.items():
    if active_keys[label]:
        ui.write(e.EV_KEY, key_code, 0)
ui.syn()

cap_video.release()
cv2.destroyAllWindows()
ui.close()
