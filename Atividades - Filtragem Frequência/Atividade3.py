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

# Definir frequência de corte para os filtros
frequencia_corte = 40

# Filtro passa-alta ideal
filtro_passa_alta_ideal = (raio > frequencia_corte).astype(float)

# Filtro passa-alta Butterworth
ordem_butterworth = 3
filtro_passa_alta_butterworth = 1 - 1 / (1 + (raio / frequencia_corte)**(2 * ordem_butterworth))

# Filtro passa-alta gaussiano
sigma = 40
filtro_passa_alta_gaussiano = 1 - np.exp(-raio**2 / (2 * (sigma**2)))

# Aplicar os filtros passa-alta ao espectro de Fourier
espectro_filtrado_passa_alta_ideal = espectro_fourier_centralizado * filtro_passa_alta_ideal
espectro_filtrado_passa_alta_butterworth = espectro_fourier_centralizado * filtro_passa_alta_butterworth
espectro_filtrado_passa_alta_gaussiano = espectro_fourier_centralizado * filtro_passa_alta_gaussiano

# Transformada inversa de Fourier para obter as imagens filtradas
imagem_filtrada_passa_alta_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_passa_alta_ideal)))
imagem_filtrada_passa_alta_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_passa_alta_butterworth)))
imagem_filtrada_passa_alta_gaussiano = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_passa_alta_gaussiano)))

# Exibir as imagens
plt.figure(figsize=(15, 12))

# Imagem original
plt.subplot(3, 4, 1)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')

# Espectro de Fourier
plt.subplot(3, 4, 2)
plt.imshow(np.log(np.abs(espectro_fourier_centralizado) + 1), cmap='gray')
plt.title('Espectro de Fourier')

# Filtro Passa-Alta Ideal
plt.subplot(3, 4, 3)
plt.imshow(filtro_passa_alta_ideal, cmap='gray')
plt.title('Filtro Passa-Alta Ideal')

# Imagem Resultante após Filtro Passa-Alta Ideal
plt.subplot(3, 4, 4)
plt.imshow(imagem_filtrada_passa_alta_ideal, cmap='gray')
plt.title('Imagem após Filtro Passa-Alta Ideal')

# Filtro Passa-Alta Butterworth
plt.subplot(3, 4, 7)
plt.imshow(filtro_passa_alta_butterworth, cmap='gray')
plt.title('Filtro Passa-Alta Butterworth')

# Imagem Resultante após Filtro Passa-Alta Butterworth
plt.subplot(3, 4, 8)
plt.imshow(imagem_filtrada_passa_alta_butterworth, cmap='gray')
plt.title('Imagem após Filtro Passa-Alta Butterworth')

# Filtro Passa-Alta Gaussiano
plt.subplot(3, 4, 11)
plt.imshow(filtro_passa_alta_gaussiano, cmap='gray')
plt.title('Filtro Passa-Alta Gaussiano')

# Imagem Resultante após Filtro Passa-Alta Gaussiano
plt.subplot(3, 4, 12)
plt.imshow(imagem_filtrada_passa_alta_gaussiano, cmap='gray')
plt.title('Imagem após Filtro Passa-Alta Gaussiano')

plt.tight_layout()
plt.show()
