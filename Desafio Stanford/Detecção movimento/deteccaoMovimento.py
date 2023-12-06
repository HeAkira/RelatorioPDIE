import cv2
import numpy as np

# Inicialize o objeto de captura de vídeo
cap = cv2.VideoCapture('output.avi')

# Inicialize o algoritmo de subtração de fundo
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aplique o algoritmo de subtração de fundo
    fgmask = fgbg.apply(frame)

    # Realize operações de limiarização ou pós-processamento, se necessário
    # Por exemplo, você pode aplicar um limiar para destacar regiões de movimento
    _, fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)

    # Encontre contornos nas regiões de movimento
    contornos, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhe os contornos na imagem original
    frame_com_contornos = frame.copy()
    cv2.drawContours(frame_com_contornos, contornos, -1, (0, 255, 0), 2)

    # Exiba a imagem resultante
    cv2.imshow('Detecção de Movimento', frame_com_contornos)

    if cv2.waitKey(30) & 0xFF == 27:  # Pressione 'Esc' para sair
        break

cap.release()
cv2.destroyAllWindows()
