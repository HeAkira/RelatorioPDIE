import cv2
import numpy as np

# Carregue a imagem
caminho_da_imagem = './morphology/noise_rectangle.tif'
imagem = cv2.imread(caminho_da_imagem, cv2.IMREAD_GRAYSCALE)

# Defina o elemento estruturante (kernel) como um quadrado 3x3
elemento_estruturante_quadrado = np.ones((3, 3), dtype=np.uint8)

# Realize a erosão para remover detalhes pequenos
erosion = cv2.erode(imagem, elemento_estruturante_quadrado, iterations=50)

# Realize a dilatação para expandir o retângulo central
dilation = cv2.dilate(erosion, elemento_estruturante_quadrado, iterations=65)

# Salvar a imagem de saída
cv2.imwrite('retangulo_central.jpg', dilation)