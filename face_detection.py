import os
import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 800) # set video width
cam.set(4, 600) # set video height
face_detector = cv2.CascadeClassifier('haarcascade_frontface.xml')

# For each person, enter one numeric face id
face_id = input('\n Input user id end press <return> ==> ')
print("\n Look your camera and wait ...")

count = 0   # count images
img_name_count = []      # read face images name

while True :
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        file_path = "Face_list/" + str(face_id) + "/"   # save image in directory
        file_list = os.listdir(file_path)
        for i in range(len(file_list)):
            img_name_count.append(int(file_list[i][2:-4]))
            count = max(img_name_count)
        count += 1
        cv2.imwrite(file_path + str(face_id) + "_" + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for stop camera
    if k == 27:
        break
    elif count % 20 == 0:  # Take 20 face sample and stop video
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()




