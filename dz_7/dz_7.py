import cv2
import numpy as np
import os
from collections import Counter

img_path = 'sar_1_gray.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Пункт 1: Сохранение монохромного изображения
np.savetxt('pixel_image.txt', img, fmt='%d')
pixel_storage_size = os.path.getsize('pixel_image.txt')


# Пункт 2: Вейвлет-преобразование Хаара
def haar_transform(image):
    rows, cols = image.shape
    image_float = image.astype(np.float64)

    row_transform = np.zeros_like(image_float)
    for i in range(rows):
        for j in range(0, cols - 1, 2):
            row_transform[i, j // 2] = (image_float[i, j] + image_float[i, j + 1]) / 2
            row_transform[i, (j // 2) + cols // 2] = (image_float[i, j] - image_float[i, j + 1]) / 2

    result = np.zeros_like(row_transform)
    for j in range(cols):
        for i in range(0, rows - 1, 2):
            result[i // 2, j] = (row_transform[i, j] + row_transform[i + 1, j]) / 2
            result[(i // 2) + rows // 2, j] = (row_transform[i, j] - row_transform[i + 1, j]) / 2

    LL = result[:rows // 2, :cols // 2]
    LH = result[:rows // 2, cols // 2:]
    HL = result[rows // 2:, :cols // 2]
    HH = result[rows // 2:, cols // 2:]

    return LL, LH, HL, HH


LL, LH, HL, HH = haar_transform(img)


# Пункт 3: Квантование высокочастотных компонент
def quantize(coeffs, n_quants):
    min_val = np.min(coeffs)
    max_val = np.max(coeffs)
    step = (max_val - min_val) / n_quants
    quantized = np.round((coeffs - min_val) / step).astype(int)
    return quantized


LH_q = quantize(LH, 4)
HL_q = quantize(HL, 4)
HH_q = quantize(HH, 4)


# Пункт 4: RLE-кодирование и сохранение
def run_length_encode(data):
    encoded = []
    for value, count in Counter(data.flatten()).items():
        encoded.append((value, count))
    return encoded


LH_rle = run_length_encode(LH_q)
HL_rle = run_length_encode(HL_q)
HH_rle = run_length_encode(HH_q)

# Сохраняем в файл в порядке LL, LH, HL, HH
with open('haar_image.txt', 'w') as f:
    np.savetxt(f, LL, fmt='%d')

    f.write("LH:\n")
    for value, count in LH_rle:
        f.write(f"{value} {count}\n")

    f.write("HL:\n")
    for value, count in HL_rle:
        f.write(f"{value} {count}\n")

    f.write("HH:\n")
    for value, count in HH_rle:
        f.write(f"{value} {count}\n")

haar_storage_size = os.path.getsize('haar_image.txt')

print(f"Исходный размер: {pixel_storage_size} байт")
print(f"Размер после сжатия: {haar_storage_size} байт")
print(f"Коэффициент сжатия: {pixel_storage_size / haar_storage_size:.2f}")