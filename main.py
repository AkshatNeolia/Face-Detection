import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)

cap = cv2.VideoCapture(0)

unrecognized_start_time = None
show_red_box = False


font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.8
thickness = 2

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    recognized_face = False

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            recognized_face = True
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            textSize = cv2.getTextSize(name, font, fontScale, thickness)[0]
            rect_x2 = x1 + textSize[0] + 10  # Adding padding
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (rect_x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 5, y2 - 10), font, fontScale, (255, 255, 255), thickness)
            markAttendance(name)

        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

            if unrecognized_start_time is None:
                unrecognized_start_time = time.time()
                show_red_box = True 

            elapsed_time = time.time() - unrecognized_start_time
            if elapsed_time > 10:
                print("No recognized face for 10 seconds. Closing...")
                cap.release()
                cv2.destroyAllWindows()
                break
            else:
                textSize = cv2.getTextSize("Unrecognized", font, fontScale, thickness)[0]
                rect_x2 = x1 + textSize[0] + 10
                cv2.rectangle(img, (x1, y2 - 35), (rect_x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "Unrecognized", (x1 + 5, y2 - 10), font, fontScale, (255, 255, 255), thickness)

    if recognized_face:
        unrecognized_start_time = None
        show_red_box = False

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
