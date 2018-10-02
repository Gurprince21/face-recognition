import cv2

face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_Cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


image = cv2.imread("guy1.jpg")

# making image gray
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_Cascade.detectMultiScale(gray_image,
scaleFactor = 1.06,
minNeighbors = 5)

# x and y is starting point of rectangle
# x+w,y+h is opposite corner of rectangle
# 0,255,0 is color of rectangle and 3 is the width
for x, y, w, h in faces:
    image = cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,0), 3)
    eyes = eye_Cascade.detectMultiScale(image, scaleFactor = 1.07)
    for ex, ey, ew, eh  in eyes:
        cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0), 2)

# resized_image = cv2.resize(image, (800, 700))

print(faces)

cv2.imshow("Face",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
