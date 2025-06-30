import cv2

# Reemplaza esta IP con la que te da la app IP Webcam

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("No se pudo acceder al stream de la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir el frame.")
        break

    cv2.imshow('Vista de la cámara IP', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
