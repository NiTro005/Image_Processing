import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

image = cv2.imread('sar_3.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Найти наиболее протяженный участок (линии Хафа)
canny = cv2.Canny(image_gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(canny, 1, np.pi / 180, threshold=150)

image_with_lines = image.copy()
max_length = 0
longest_line = None

if lines is not None:
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho

        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
        length = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

        if length > max_length:
            max_length = length
            longest_line = (rho, theta, pt1, pt2)

if longest_line is not None:
    rho, theta, pt1, pt2 = longest_line
    cv2.line(image_with_lines, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

# 2. Исследование алгоритмов бинаризации для выделения дорожной полосы
_, thresh_otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_adaptive = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 51, 10)

road_mask = thresh_adaptive == 0
image_road = image.copy()
image_road[road_mask] = [0, 255, 0]


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Наиболее протяженная линия (Хаф)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(thresh_otsu, cmap='gray')
plt.title('Бинаризация: Метод Оцу')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image_road, cv2.COLOR_BGR2RGB))
plt.title('Выделенная дорожная полоса')
plt.axis('off')

plt.tight_layout()
plt.show()