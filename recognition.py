import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'			# Directorio de fotos conocidas
UNKNOWN_FACES_DIR = 'unknown_faces'		# Directorio con fotos sin identificar
TOLERANCE = 0.5							# Entre más alto, más prob. de falsos positivos (0.6 recomendado)
FRAME_THICKNESS = 3
FONT_THICKNES = 2
MODEL = 'cnn'							# To use a pre-trained convolutional neural network.

# Iterate over known images/faces
print('Cargando imágenes con caras conocidas' + '\n')

known_faces = []						# Lista con datos de las caras conocidas
known_names = []						# Nombres asociados a las caras conocidas

for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):	# Lista los archivos de las carpetas en KNOWN_FACES_DIR
		image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')	# Leer cada imagen en la carpeta
		encoding = face_recognition.face_encodings(image)[0]								# Image enconding de la cara identificada
		known_faces.append(encoding)		# Llenar la lista known_faces con lo encodings de las caras identificadas
		known_names.append(name)			# Llenar la lista known_names con el nombre de la carpeta donde están las caras conocidas

# Iterate over the unknown images, find all faces and compare them to the known faces
print('Procesando imágenes con caras desconocidas:')
for filename in os.listdir(UNKNOWN_FACES_DIR):
	print('\n' + filename)
	image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')		# Cargar imagen
	locations = face_recognition.face_locations(image, model=MODEL)		# Localicar todas las caras que aparezcan en las fotos 
	encodings = face_recognition.face_encodings(image, locations)		# Registrar caras encontradas con face_locations
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)	# Convertir image de RGB a BGR para ser compatible con opencv-python

	# Iterate over the encodings and locations
	for face_enconding, face_location in zip(encodings, locations):		# zip() Function for Parallel Iteration
		results = face_recognition.compare_faces(known_faces, face_enconding, TOLERANCE)	# Compara caras conocidas con las registradas

		match = None													
		if True in results:
			match = known_names[results.index(True)]
			print(f'Rostro reconocido: {match}')

			# Puntos para graficar el recuadro en la cara encontrada
			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1], face_location[2])
			color = [0, 255, 0]		# Green color
			cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

			# Recuadro con etiqueta de nombre
			top_left = (face_location[3], face_location[2])
			bottom_right = (face_location[1], face_location[2]+22)
			cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
			cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), FONT_THICKNES)

	cv2.imshow(filename, image)
	cv2.waitKey(1000)
	#cv2.destroyWindow(filename)