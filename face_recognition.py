import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0 # count id
# change names list, if you want
names = ['Joohyeong', 'Jeawon', 'jeeyoung']

cam = cv2.VideoCapture(0)
cam.set(3, 800)  # set video widht
cam.set(4, 600)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id_S, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if confidence < 100:
            id_s = names[id]
            confidence = "  {0}%".format(round(100 - confidence))

        else:
            id_s = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id_s), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Face recognition', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
