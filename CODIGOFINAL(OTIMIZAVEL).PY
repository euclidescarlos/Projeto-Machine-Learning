# ...existing code...
import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import winsound

# Configurações
MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"
LOG_FILE = "registro_entradas_saidas.txt"

ENTRY_SECONDS = 10.0  # segundos contínuos para registrar entrada
EXIT_SECONDS = 10.0   # segundos contínuos para registrar saída
CONFIDENCE_THRESHOLD = 0.8
ALERT_SECONDS = 60.0  # tempo para disparar alerta
ALERT_INTERVAL = 5.0  # intervalo do beep

# Carregar modelo
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Carregar labels
with open(LABELS_PATH, "r") as f:
    labels = [l.strip() for l in f.readlines()]

# Inicializar câmera e detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
people_tracking = {}

if not cap.isOpened():
    print("Erro ao acessar a câmera")
    exit()
else:
    print("Câmera acessada com sucesso. Pressione 'q' para sair.")

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
            face = frame[y:y+h, x:x+w]
            img = cv2.resize(face, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = (img.astype(np.float32) / 127.5) - 1

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            index = int(np.argmax(prediction))
            confidence = float(prediction[0][index])
            label = labels[index] if 0 <= index < len(labels) else "Desconhecido"

            if confidence >= CONFIDENCE_THRESHOLD:
                detected_in_frame.add(label)
                if label not in people_tracking:
                    people_tracking[label] = {
                        'present': False,
                        'entry_time': None,
                        'seen_start': None,
                        'leave_start': None,
                        'absent_since': None,
                        'last_alert': 0,
                        'last_bbox': None
                    }
                person = people_tracking[label]

                # atualizar bbox conhecido
                person['last_bbox'] = (x, y, w, h)

                if not person['present']:
                    # contagem para ENTRADA enquanto é detectado continuamente
                    if person['seen_start'] is None:
                        person['seen_start'] = current_time
                    seen_time = current_time - person['seen_start']
                    # Barra de progresso (entrada)
                    progress = min(int((seen_time / ENTRY_SECONDS) * w), w)
                    cv2.rectangle(frame, (x, y-20), (x + progress, y-10), (0, 255, 0), -1)
                    cv2.rectangle(frame, (x, y-20), (x + w, y-10), (255, 255, 255), 1)
                    cv2.putText(frame, f"{min(int(seen_time), int(ENTRY_SECONDS))}s/{int(ENTRY_SECONDS)}s",
                                (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    if seen_time >= ENTRY_SECONDS:
                        person['present'] = True
                        person['entry_time'] = current_time
                        person['seen_start'] = None
                        person['absent_since'] = None
                        person['leave_start'] = None
                        event_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        log_file.write(f"{event_time} - {label}: ENTRADA\n")
                        log_file.flush()
                        print(f"{event_time} - ENTRADA registrado para {label}")
                else:
                    # pessoa já marcada como presente e foi detectada agora
                    if person['absent_since'] is not None:
                        # iniciar contagem para SAÍDA quando reaparecer
                        if person['leave_start'] is None:
                            person['leave_start'] = current_time
                        leave_time = current_time - person['leave_start']
                        # Barra de progresso (saída) desenhada em vermelho sobre bbox atual
                        progress = min(int((leave_time / EXIT_SECONDS) * w), w)
                        cv2.rectangle(frame, (x, y-20), (x + progress, y-10), (0, 0, 255), -1)
                        cv2.rectangle(frame, (x, y-20), (x + w, y-10), (255, 255, 255), 1)
                        cv2.putText(frame, f"{min(int(leave_time), int(EXIT_SECONDS))}s/{int(EXIT_SECONDS)}s",
                                    (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        if leave_time >= EXIT_SECONDS:
                            # Registrar SAÍDA
                            total_time = current_time - (person['entry_time'] or current_time)
                            event_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            log_file.write(f"{event_time} - {label}: SAÍDA (Total: {total_time:.2f}s)\n")
                            log_file.flush()
                            print(f"{event_time} - SAÍDA registrado para {label} (tempo total: {total_time:.2f}s)")
                            person['present'] = False
                            person['entry_time'] = None
                            person['seen_start'] = None
                            person['leave_start'] = None
                            person['absent_since'] = None
                            person['last_alert'] = 0
                            person['last_bbox'] = None
                            # não continuar desenhando a bbox/label após saída
                            continue
                    else:
                        # detectado normalmente enquanto presente (sem ter ficado ausente)
                        person['leave_start'] = None
                        person['absent_since'] = None
                        total_time = current_time - (person['entry_time'] or current_time)
                        cv2.putText(frame, f"{label} (presente) - {int(total_time)}s no local",
                                    (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Desenho do rosto e label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Desconhecido", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Pós-processamento: para cada pessoa, cuidar de ausências, SAÍDA (via re-enquadramento) e ALERTA contínuo
        for name, pdata in list(people_tracking.items()):
            if name not in detected_in_frame:
                # se ausente no frame
                if pdata['present']:
                    # marcar quando ficou ausente (usar para detectar reaparecimento)
                    if pdata['absent_since'] is None:
                        pdata['absent_since'] = current_time
                    # enquanto ausente, cancelar qualquer contagem de leave (será iniciada quando reaparecer)
                    pdata['leave_start'] = None
                    # manter last_bbox (pode limpar se preferir)
                else:
                    # não presente e não detectado: resetar contadores de visão e bbox
                    pdata['seen_start'] = None
                    pdata['last_bbox'] = None
            else:
                # detectado no frame -> nada a fazer aqui além do que já foi feito no loop de faces
                pass

            # ALERTA: emitir beep e log para quem está marcado como presente,
            # independentemente de estar no frame ou não
            if pdata.get('present'):
                total_time = current_time - (pdata['entry_time'] or current_time)
                if total_time >= ALERT_SECONDS:
                    if current_time - pdata.get('last_alert', 0) >= ALERT_INTERVAL:
                        print(f"ALERTA: Tempo excedido para {name}")
                        try:
                            winsound.Beep(1000, 500)  # beep de 0,5s
                        except Exception:
                            # em ambientes sem winsound disponível, apenas ignore o beep
                            pass
                        event_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        log_file.write(f"{event_time} - ALERTA: Tempo excedido para {name}\n")
                        log_file.flush()
                        pdata['last_alert'] = current_time

        cv2.imshow('Reconhecimento Facial (Teachable Machine)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
# ...existing code...