import cv2
import numpy as np

# Carregue a imagem
caminho_da_imagem = './morphology/text_gaps.tif'
imagem = cv2.imread(caminho_da_imagem, cv2.IMREAD_GRAYSCALE)

# Defina os elementos estruturantes (kernel)
# Elemento estruturante em forma de cruz:
elemento_estruturante_cruz = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]], dtype=np.uint8)

# Elemento estruturante em forma de quadrado:
elemento_estruturante_quadrado = np.ones((3, 3), dtype=np.uint8)

# Elemento estruturante em forma de coluna (7 pixels de comprimento):
elemento_estruturante_coluna = np.ones((7, 1), dtype=np.uint8)

# Elemento estruturante em forma de losango:
elemento_estruturante_losango = np.array([[0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 1, 1, 1, 0, 0],
                                       [0, 1, 1, 1, 1, 1, 0],
                                       [1, 1, 1, 1, 1, 1, 1],
                                       [0, 1, 1, 1, 1, 1, 0],
                                       [0, 0, 1, 1, 1, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)

# Aplicar erosão e dilatação com os elementos estruturantes definidos
erosion_cruz = cv2.erode(imagem, elemento_estruturante_cruz, iterations=1)
dilation_cruz = cv2.dilate(imagem, elemento_estruturante_cruz, iterations=1)

erosion_quadrado = cv2.erode(imagem, elemento_estruturante_quadrado, iterations=1)
dilation_quadrado = cv2.dilate(imagem, elemento_estruturante_quadrado, iterations=1)

erosion_coluna = cv2.erode(imagem, elemento_estruturante_coluna, iterations=1)
dilation_coluna = cv2.dilate(imagem, elemento_estruturante_coluna, iterations=1)

erosion_losango = cv2.erode(imagem, elemento_estruturante_losango, iterations=1)
dilation_losango = cv2.dilate(imagem, elemento_estruturante_losango, iterations=1)

# Salvar as imagens de saída
cv2.imwrite('erosion_cruz_output.jpg', erosion_cruz)
cv2.imwrite('dilation_cruz_output.jpg', dilation_cruz)

cv2.imwrite('erosion_quadrado_output.jpg', erosion_quadrado)
cv2.imwrite('dilation_quadrado_output.jpg', dilation_quadrado)

cv2.imwrite('erosion_coluna_output.jpg', erosion_coluna)
cv2.imwrite('dilation_coluna_output.jpg', dilation_coluna)

cv2.imwrite('erosion_losango_output.jpg', erosion_losango)
cv2.imwrite('dilation_losango_output.jpg', dilation_losango)