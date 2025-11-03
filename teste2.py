import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime


# algumas configurações importantes
MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"
LOG_FILE = "registro_entradas_saidas.txt"

ENTRY_SECONDS = 10.0  # segundos contínuos para registrar entrada
EXIT_SECONDS = 10.0   # segundos contínuos para registrar saída (quando a pessoa reaparece para sair)
CONFIDENCE_THRESHOLD = 0.8

# carregar modelo TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# carregando labels
with open(LABELS_PATH, "r") as f:
    labels = [l.strip() for l in f.readlines()]

# inicializar camera e detector de faces
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
        detected_in_frame = set()  # nomes detectados neste frame

        # para cada rosto detectado, predição e controle de timers
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            img = cv2.resize(face, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = (img.astype(np.float32) / 127.5) - 1  # normalização padrão TM

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            index = int(np.argmax(prediction))
            confidence = float(prediction[0][index])
            label = labels[index] if 0 <= index < len(labels) else "Desconhecido"

            # considerando detencções confiaveis
            if confidence >= CONFIDENCE_THRESHOLD:
                detected_in_frame.add(label)

                # garantir que a pessoa existe no dicionário
                if label not in people_tracking:
                    people_tracking[label] = {
                        'present': False,
                        'entry_time': None,
                        'continuous_start': None
                    }

                person = people_tracking[label]

                # se acabou de aparecer (continuous_start é None), iniciar contador contínuo
                if person['continuous_start'] is None:
                    person['continuous_start'] = current_time

                # tempo contínuo atual
                continuous_time = current_time - person['continuous_start']

                # desenhos UI: barra de progresso para os 10s
                progress = min(int((continuous_time / ENTRY_SECONDS) * w), w)
                cv2.rectangle(frame, (x, y-20), (x + progress, y-10), (0, 255, 0), -1)
                cv2.rectangle(frame, (x, y-20), (x + w, y-10), (255, 255, 255), 1)
                cv2.putText(frame, f"{min(int(continuous_time), int(ENTRY_SECONDS))}s/{int(ENTRY_SECONDS)}s",
                            (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # lógica de entrada/saída baseada em estado atual
                if not person['present']:
                    # ainda não entrou hoje: precisa de ENTRY_SECONDS contínuos para registrar ENTRADA
                    if continuous_time >= ENTRY_SECONDS:
                        person['present'] = True
                        person['entry_time'] = current_time
                        person['continuous_start'] = None  # zera contador para próxima ação
                        event_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        log_file.write(f"{event_time} - {label}: ENTRADA\n")
                        log_file.flush()
                        print(f"{event_time} - ENTRADA registrado para {label}")
                else:
                    # já está presente (registrou ENTRADA). Para registrar SAÍDA, a pessoa deve
                    # reaparecer e ficar ENTRY_SECONDS contínuos — usamos EXIT_SECONDS para simetria.
                    # aqui continuous_time é o tempo desde que reapareceu.
                    cv2.putText(frame, f"{label} (presente) - {int(current_time - person['entry_time'])}s no local",
                                (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    if continuous_time >= EXIT_SECONDS:
                        # registrar SAÍDA
                        total_time = current_time - person['entry_time'] if person['entry_time'] else 0
                        event_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        log_file.write(f"{event_time} - {label}: SAÍDA (Total: {total_time:.2f}s)\n")
                        log_file.flush()
                        print(f"{event_time} - SAÍDA registrado para {label} (tempo total: {total_time:.2f}s)")
                        # resetar estado para person ficar 'ausente' até nova ENTRADA
                        person['present'] = False
                        person['entry_time'] = None
                        person['continuous_start'] = None

                # desenho do retângulo e nome
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # baixa confiança -> tratar como não-detectado (mostra vermelho)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Desconhecido", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # IMPORTANTE: resetar continuous_start para usuários que NÃO foram detectados neste frame
        # Isso garante que o tempo exige CONTINUIDADE (qualquer ausente reseta).
        for name, pdata in people_tracking.items():
            if name not in detected_in_frame:
                # Ao sumir do frame, interrompe qualquer contagem contínua em andamento
                pdata['continuous_start'] = None

        # mostrar vídeo
        cv2.imshow('Reconhecimento Facial (Teachable Machine)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
