# Importar múdulos de reconocimiento facial y OpenCV-Python
import face_recognition
import cv2				

# Cargar la imagen en un arreglo de numpy (matrices)
image = face_recognition.load_image_file("faces.jpg")
imageBGR = cv2.imread("faces.jpg")

'''	Encontrar las caras en la imagen
	usando un modelo de redes neuronales convolucionales '''
face_locations = face_recognition.face_locations(image, model="cnn")

print(f"I found {len(face_locations)} face(s) in this photograph.")

for face_location in face_locations:
	# Imprimir coordenadas de la cara encontrada
    top, right, bottom, left = face_location
    print(f"A face is located at pixel location Top:{top}, Left: {left}, Bottom: {bottom}, Right: {right}")

    # Graficar un rectángulo en cada cara
    im = cv2.rectangle(imageBGR, (left, top), (right, bottom), (0, 255, 0), 2)

    # Imprimir la imagen 
    cv2.imshow('Im', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()