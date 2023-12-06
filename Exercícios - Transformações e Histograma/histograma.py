# Importar as bibliotecas necessárias
import cv2
import numpy as np

# Ler a imagem
img = cv2.imread('fractured_spine.jpg')

# Converter a imagem para o espaço de cores YUV
img_para_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# Realizar equalização de histograma no canal Y (luminância)
img_para_yuv[:, :, 0] = cv2.equalizeHist(img_para_yuv[:, :, 0])

# Converter a imagem resultante de volta para o espaço de cores BGR
resultado_equalizacao_hist = cv2.cvtColor(img_para_yuv, cv2.COLOR_YUV2BGR)

# Salvar a imagem resultante da equalização de histograma
cv2.imwrite('resultado.jpg', resultado_equalizacao_hist)