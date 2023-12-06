import cv2
import numpy as np

# Carregue a imagem
caminho_da_imagem = './morphology/text_gaps.tif'
imagem = cv2.imread(caminho_da_imagem, cv2.IMREAD_GRAYSCALE)

# Defina o elemento estruturante (kernel) como uma cruz
elemento_estruturante_cruz = np.array([[0, 1, 0],
                                           [1, 1, 1],
                                           [0, 1, 0],], dtype=np.uint8)

# Realize a operação de dilatação para melhorar a qualidade da imagem
dilation = cv2.dilate(imagem, elemento_estruturante_cruz, iterations=1)

# Realize a erosão para remover detalhes pequenos
erosion = cv2.erode(dilation, elemento_estruturante_cruz, iterations=1)

# Salvar a imagem de saída
cv2.imwrite('imagem_melhorada.jpg', erosion)