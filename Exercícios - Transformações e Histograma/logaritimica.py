# Importar as bibliotecas necessárias
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ler uma imagem
imagem = cv2.imread('fractured_spine.jpg')

# Aplicar o método de transformação logarítmica
imagem_log = 150 * (np.log(imagem + 1))

# Especificar o tipo de dados para que
# os valores float sejam convertidos para inteiros
imagem_log = np.array(imagem_log, dtype=np.uint8)

# Exibir ambas as imagens
plt.imshow(imagem)
plt.show()
plt.imshow(imagem_log)
plt.show()