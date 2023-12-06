import cv2
import numpy as np

# Carregue a imagem
caminho_da_imagem = './morphology/rosto_perfil.tif'
imagem = cv2.imread(caminho_da_imagem, cv2.IMREAD_GRAYSCALE)

# Defina o elemento estruturante (kernel) como um quadrado 3x3
elemento_estruturante_quadrado = np.ones((3, 3), dtype=np.uint8)

# Realize a operação de gradiente morfológico
gradiente_morfologico = cv2.morphologyEx(imagem, cv2.MORPH_GRADIENT, elemento_estruturante_quadrado)

# Salvar a imagem de saída
cv2.imwrite('borda_da_imagem.jpg', gradiente_morfologico)