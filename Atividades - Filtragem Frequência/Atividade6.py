import numpy as np
import matplotlib.pyplot as plt
import cv2

# Carregar a imagem
imagem = cv2.imread('car.tif', cv2.IMREAD_GRAYSCALE)

# Calcular o espectro de Fourier da imagem
espectro_fourier = np.fft.fft2(imagem)
espectro_fourier_centralizado = np.fft.fftshift(espectro_fourier)

# Dimensões da imagem
altura, largura = imagem.shape

# Criar grades de frequência para a aplicação dos filtros
x, y = np.meshgrid(np.arange(-largura/2, largura/2), np.arange(-altura/2, altura/2))
raio = np.sqrt(x**2 + y**2)

# Definir as frequências de corte para o filtro passa-banda
frequencia_corte_inferior = 0.3
frequencia_corte_superior = 0.9

# Filtro passa-alta para realçar frequências acima da frequência de corte inferior
filtro_passa_alta = (raio > frequencia_corte_inferior).astype(float)

# Filtro passa-baixa para eliminar frequências acima da frequência de corte superior
filtro_passa_baixa = (raio <= frequencia_corte_superior).astype(float)

# Combinar os filtros passa-alta e passa-baixa para criar o filtro passa-banda
filtro_passa_banda = filtro_passa_alta * filtro_passa_baixa

# Aplicar o filtro passa-banda ao espectro de Fourier
espectro_filtrado_passa_banda = espectro_fourier_centralizado * filtro_passa_banda

# Transformada inversa de Fourier para obter a imagem filtrada
imagem_filtrada_passa_banda = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_passa_banda)))

# Exibir as imagens
plt.figure(figsize=(12, 8))

# Imagem original
plt.subplot(2, 3, 1)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')

# Espectro de Fourier
plt.subplot(2, 3, 2)
plt.imshow(np.log(np.abs(espectro_fourier_centralizado) + 1), cmap='gray')
plt.title('Espectro de Fourier')

# Filtro Passa-Banda
plt.subplot(2, 3, 3)
plt.imshow(filtro_passa_banda, cmap='gray')
plt.title('Filtro Passa-Banda')

# Imagem Resultante após Filtro Passa-Banda
plt.subplot(2, 3, 4)
plt.imshow(imagem_filtrada_passa_banda, cmap='gray')
plt.title('Imagem após Filtro Passa-Banda')

plt.tight_layout()
plt.show()