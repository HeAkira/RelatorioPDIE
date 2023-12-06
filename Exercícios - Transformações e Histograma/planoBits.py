# Importar as bibliotecas necessárias
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

# Ler a imagem em tons de cinza
img = cv2.imread('fractured_spine.jpg', 0)

# Iterar sobre cada pixel e converter o valor do pixel para binário usando np.binary_repr() e armazená-lo em uma lista.
lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
         lst.append(np.binary_repr(img[i][j], width=8))  # width = número de bits

# Temos uma lista de strings em que cada string representa o valor binário de um pixel. Para extrair os planos de bits, precisamos iterar sobre as strings e armazenar os caracteres correspondentes aos planos de bits em listas.
# Multiplicar por 2^(n-1) e remodelar para reconstruir a imagem de bits.
imagem_oito_bits = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(img.shape[0], img.shape[1])
imagem_sete_bits = (np.array([int(i[1]) for i in lst], dtype=np.uint8) * 64).reshape(img.shape[0], img.shape[1])
imagem_seis_bits = (np.array([int(i[2]) for i in lst], dtype=np.uint8) * 32).reshape(img.shape[0], img.shape[1])
imagem_cinco_bits = (np.array([int(i[3]) for i in lst], dtype=np.uint8) * 16).reshape(img.shape[0], img.shape[1])
imagem_quatro_bits = (np.array([int(i[4]) for i in lst], dtype=np.uint8) * 8).reshape(img.shape[0], img.shape[1])
imagem_tres_bits = (np.array([int(i[5]) for i in lst], dtype=np.uint8) * 4).reshape(img.shape[0], img.shape[1])
imagem_dois_bits = (np.array([int(i[6]) for i in lst], dtype=np.uint8) * 2).reshape(img.shape[0], img.shape[1])
imagem_um_bit = (np.array([int(i[7]) for i in lst], dtype=np.uint8) * 1).reshape(img.shape[0], img.shape[1])

# Concatenar essas imagens para facilitar a exibição usando cv2.hconcat()
finalr = cv2.hconcat([imagem_oito_bits, imagem_sete_bits, imagem_seis_bits, imagem_cinco_bits])
finalv = cv2.hconcat([imagem_quatro_bits, imagem_tres_bits, imagem_dois_bits, imagem_um_bit])

# Concatenar verticalmente
final = cv2.hconcat([finalr, finalv])

# Preparar a janela do Matplotlib para plotar as imagens
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.5)

axes[0, 0].imshow(imagem_oito_bits)
axes[0, 0].set_title('Plano de bits 8')
axes[0, 1].imshow(imagem_sete_bits)
axes[0, 1].set_title('Plano de bits 7')
axes[0, 2].imshow(imagem_seis_bits)
axes[0, 2].set_title('Plano de bits 6')
axes[0, 3].imshow(imagem_cinco_bits)
axes[0, 3].set_title('Plano de bits 5')
axes[1, 0].imshow(imagem_quatro_bits)
axes[1, 0].set_title('Plano de bits 4')
axes[1, 1].imshow(imagem_tres_bits)
axes[1, 1].set_title('Plano de bits 3')
axes[1, 2].imshow(imagem_dois_bits)
axes[1, 2].set_title('Plano de bits 2')
axes[1, 3].imshow(imagem_um_bit)
axes[1, 3].set_title('Plano de bits 1')
plt.show()

# Combinar 4 planos de bits
nova_imagem = imagem_oito_bits + imagem_sete_bits + imagem_seis_bits + imagem_cinco_bits

# Exibir a imagem combinada
plt.imshow(nova_imagem)
plt.show()