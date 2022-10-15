import cv2
import face_recognition

imgBill = face_recognition.load_image_file('images_sample/billgates.jpg')
imgBill = cv2.cvtColor(imgBill, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images_sample/billgates_test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
width, height = 400, 400
imgBill = cv2.resize(imgBill, (width, height))
imgTest = cv2.resize(imgTest, (width, height))

faceloc = face_recognition.face_locations(imgBill)[0]
encodeBill = face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill,
              (faceloc[3], faceloc[0]),
              (faceloc[1], faceloc[2]),
              (255, 0, 255),
              2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,
              (facelocTest[3], facelocTest[0]),
              (facelocTest[1], facelocTest[2]),
              (255, 0, 255),
              3)

results = face_recognition.compare_faces([encodeBill], encodeTest)
faceDis = face_recognition.face_distance([encodeBill], encodeTest)
print(results, faceDis)
cv2.putText(imgTest,
            f'{results}{round(faceDis[0], 2)}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
            2)

cv2.imshow('billgates', imgBill)
cv2.imshow('Billgates_test', imgTest)
cv2.waitKey(0)
