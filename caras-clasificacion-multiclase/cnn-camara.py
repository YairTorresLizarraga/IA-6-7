import cv2
import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model('clasificacion.keras')
class_names = sorted(
    [
        name
        for name in os.listdir('dataset-procesado')
        if os.path.isdir(os.path.join('dataset-procesado', name))
    ]
)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

while True:
    ok, frame = camera.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rostro_recortado = frame[y:y + h, x:x + w]
        rostro_rgb = cv2.cvtColor(rostro_recortado, cv2.COLOR_BGR2RGB)
        rostro_final = cv2.resize(rostro_rgb, (60, 60), interpolation=cv2.INTER_CUBIC)

        input_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(rostro_final)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        probs = model.predict(input_tensor, verbose=0)[0]

        best_index = int(np.argmax(probs))
        confidence = float(probs[best_index])

        cv2.putText(
            frame,
            f'{class_names[best_index]} ({confidence:.2%})',
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow('clasificacion multiclase', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
