import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import winsound

# ---------------- CONFIGURAÇÕES ----------------
MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"
LOG_FILE = "registro_entradas_saidas.txt"

ENTRY_SECONDS = 10.0   # segundos contínuos para registrar entrada
EXIT_SECONDS = 10.0    # segundos contínuos para registrar saída
CONFIDENCE_THRESHOLD = 0.85
ALERT_SECONDS = 60.0   # tempo para disparar alerta
ALERT_INTERVAL = 5.0   # intervalo do beep

# ---------------- CARREGAR MODELO ----------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    labels = [l.strip() for l in f.readlines()]

# ---------------- INICIALIZAÇÃO ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
people_tracking = {}

if not cap.isOpened():
    print("Erro ao acessar a câmera")
    exit()
else:
    print("Câmera acessada com sucesso. Pressione 'q' para sair.")

# ---------------- FUNÇÕES ----------------

def log_event(log_file, label, event):
    event_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file.write(f"{event_time} - {label}: {event}\n")
    log_file.flush()
    print(f"{event_time} - {label}: {event}")

def process_face(face_img):
    img = cv2.resize(face_img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = (img.astype(np.float32)/127.5) - 1
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index])
    label = labels[index] if 0 <= index < len(labels) else "Desconhecido"
    return label, confidence

def draw_bar(frame, x, y, w, progress, color):
    cv2.rectangle(frame, (x, y-20), (x + progress, y-10), color, -1)
    cv2.rectangle(frame, (x, y-20), (x + w, y-10), (255, 255, 255), 1)

def draw_face(frame, bbox, label, confidence, recognized=True):
    x, y, w, h = bbox
    color = (0, 255, 0) if recognized else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    text_y = max(y-35,0)
    cv2.putText(frame, f"{label} ({confidence:.2f})" if recognized else "Desconhecido", 
                (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def handle_entry(person, current_time, bbox, frame, log_file):
    x, y, w, h = bbox
    if person['seen_start'] is None:
        person['seen_start'] = current_time
    seen_time = current_time - person['seen_start']
    bar_width = min(int((seen_time / ENTRY_SECONDS) * w), w)
    draw_bar(frame, x, y, w, bar_width, (0, 255, 0))
    cv2.putText(frame, f"{min(int(seen_time), int(ENTRY_SECONDS))}s/{int(ENTRY_SECONDS)}s",
                (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if seen_time >= ENTRY_SECONDS:
        person['present'] = True
        person['entry_time'] = current_time
        person['seen_start'] = None
        person['absent_since'] = None
        person['leave_start'] = None
        log_event(log_file, person['label'], "ENTRADA")

def handle_exit(person, current_time, bbox, frame, log_file):
    x, y, w, h = bbox
    if person['leave_start'] is None:
        person['leave_start'] = current_time
    leave_time = current_time - person['leave_start']
    bar_width = min(int((leave_time / EXIT_SECONDS) * w), w)
    draw_bar(frame, x, y, w, bar_width, (0, 0, 255))
    cv2.putText(frame, f"{min(int(leave_time), int(EXIT_SECONDS))}s/{int(EXIT_SECONDS)}s",
                (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if leave_time >= EXIT_SECONDS:
        total_time = current_time - (person['entry_time'] or current_time)
        log_event(log_file, person['label'], f"SAÍDA (Total: {total_time:.2f}s)")
        # resetar
        for key in ['present','entry_time','seen_start','leave_start','absent_since','last_alert']:
            person[key] = None
        person['present'] = False

def handle_alert(person, current_time, log_file):
    if person['present']:
        total_time = current_time - (person['entry_time'] or current_time)
        if total_time >= ALERT_SECONDS:
            if current_time - person.get('last_alert',0) >= ALERT_INTERVAL:
                print(f"ALERTA: Tempo excedido para {person['label']}")
                try:
                    winsound.Beep(1000,500)
                except Exception:
                    pass
                log_event(log_file, person['label'], "ALERTA: Tempo excedido")
                person['last_alert'] = current_time

# ---------------- LOOP PRINCIPAL ----------------
with open(LOG_FILE, 'a') as log_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível receber o frame. Saindo ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        current_time = time.time()
        detected_in_frame = set()

        for (x, y, w, h) in faces:
            bbox = (x, y, w, h)
            face = frame[y:y+h, x:x+w]
            label, confidence = process_face(face)

            recognized = confidence >= CONFIDENCE_THRESHOLD
            detected_in_frame.add(label) if recognized else None

            if label not in people_tracking and recognized:
                people_tracking[label] = {
                    'label': label,
                    'present': False,
                    'entry_time': None,
                    'seen_start': None,
                    'leave_start': None,
                    'absent_since': None,
                    'last_alert': 0,
                    'last_bbox': None
                }

            if recognized:
                person = people_tracking[label]
                person['last_bbox'] = bbox

                if not person['present']:
                    handle_entry(person, current_time, bbox, frame, log_file)
                else:
                    total_time = current_time - (person['entry_time'] or current_time)
                    cv2.putText(frame, f"{label} (presente) - {int(total_time)}s no local",
                                (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    if person['absent_since'] is not None:
                        handle_exit(person, current_time, bbox, frame, log_file)
                    handle_alert(person, current_time, log_file)

            draw_face(frame, bbox, label, confidence, recognized)

        # Pós-processamento para ausentes
        for name, pdata in list(people_tracking.items()):
            if name not in detected_in_frame:
                if pdata['present']:
                    if pdata['absent_since'] is None:
                        pdata['absent_since'] = current_time
                    pdata['leave_start'] = None
                else:
                    pdata['seen_start'] = None
                    pdata['last_bbox'] = None

        # ALERTA global (para todos presentes, mesmo fora do frame)
        for pdata in people_tracking.values():
            handle_alert(pdata, current_time, log_file)

        cv2.imshow('Reconhecimento Facial (Teachable Machine)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
