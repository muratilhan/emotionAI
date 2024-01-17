import cv2
import tensorflow as tf
import numpy as np

recording=False
def emotion_recognition():
    model = tf.keras.models.load_model('duygu_tanima_modeli_v4.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    emotions = ["Kizgin", "Igrenmis", "Korkmus", "Mutlu", "Notr", "Mutsuz", "Sasirmis"]
    emotion_counts = {emotion: 0 for emotion in emotions}

    cap = cv2.VideoCapture(0)

    while True:
        print(recording)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.reshape(1, 48, 48, 1)

            prediction = model.predict(face)
            emotion_index = np.argmax(prediction)
            emotion_name = emotions[emotion_index]

            emotion_counts[emotion_name] += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Duygu Tanıma', frame)   
        if(recording):
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    most_common_emotion = max(emotion_counts, key=emotion_counts.get)
    result = {'commonEmotion': most_common_emotion, 'emotionCounts': emotion_counts}
    return result

