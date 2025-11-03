import cv2
import numpy as np
import tensorflow as tf

# recuperar arquivo do Teachanable Machine:
# estamos utilizando a versão TensorFlow Lite (TensorFlow normal estava dando erro)
# deixamos o arquivo tensorflow keras_model.h5 e o tensorflow lite model_unquant.tflite no Notebook.
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
labels = open("labels.txt", "r").readlines()
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# carregando o classificador e inicializando a câmera
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera")
else:
    print("Câmera acessada com sucesso. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível receber o frame. Saindo ...")
            break

        # aqui estamos convertendo a escala para cinza para que o haar (classificador) funcione melhor
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # inicia a detecção de rostos e começa a tentar assimilar, redimensionando e fazendo a predição.
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            img = cv2.resize(face, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = (img.astype(np.float32) / 127.5) - 1  # normalização padrão do Teachable Machine

            # parte de predição
            img = cv2.resize(frame, (224, 224))  # o tamanho depende do modelo do Teachable Machine
            img = np.expand_dims(img, axis=0)
            img = (img.astype(np.float32) / 127.5) - 1  # normalização padrão TM  
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            index = np.argmax(prediction)
            confidence = prediction[0][index]
            label = labels[index].strip()

            color = (0, 255, 0) if confidence > 0.8 else (0, 0, 255)  # desenhando um retângulo ao redor de cada rosto
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Reconhecimento Facial (Teachable Machine)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()