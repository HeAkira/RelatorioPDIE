import cv2
import numpy as np

# Carregue as imagens da placa sem defeito e da placa com defeito
placa_sem_defeito = cv2.imread('pcbCroppedTranslated.png', cv2.IMREAD_GRAYSCALE)
placa_com_defeito = cv2.imread('pcbCroppedTranslatedDefected.png', cv2.IMREAD_GRAYSCALE)

# Verifique se as imagens foram carregadas corretamente
if placa_sem_defeito is None or placa_com_defeito is None:
    print("Erro ao carregar as imagens.")
else:
    # Realize a subtração entre as duas imagens
    diferenca = cv2.absdiff(placa_sem_defeito, placa_com_defeito)

    # Aplique um limiar para destacar as diferenças
    limiar = 30
    _, diferenca_binaria = cv2.threshold(diferenca, limiar, 255, cv2.THRESH_BINARY)

    # Encontre contornos nas diferenças destacadas
    contornos, _ = cv2.findContours(diferenca_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhe os contornos nas imagens originais
    resultado = placa_com_defeito.copy()
    cv2.drawContours(resultado, contornos, -1, (0, 0, 255), 2)

    # Exiba a imagem com os contornos destacados
    cv2.imshow('Resultado', resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()