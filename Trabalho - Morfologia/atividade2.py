import cv2
import numpy as np

# Carregue a imagem
caminho_da_imagem = './morphology/morfologia2.tif'
imagem = cv2.imread(caminho_da_imagem, cv2.IMREAD_GRAYSCALE)

# Defina o elemento estruturante em forma de cruz
elemento_estruturante_cruz = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0],], dtype=np.uint8)

# Realize a operação de abertura (erosão seguida de dilatação)
abertura_cruz = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, elemento_estruturante_cruz)

# Realize a operação de fechamento (dilatação seguida de erosão)
fechamento_cruz = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, elemento_estruturante_cruz)

# Salvar as imagens de saída
cv2.imwrite('abertura_cruz_output.jpg', abertura_cruz)
cv2.imwrite('fechamento_cruz_output.jpg', fechamento_cruz)