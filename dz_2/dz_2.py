import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error

# Загрузка изображения
image = cv2.imread('sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("=== ДОМАШНЕЕ ЗАДАНИЕ 2 ===")

# 1. Зашумить изображение при помощи шума гаусса, постоянного шума
print("1. Добавление шумов:")

# Гауссов шум
mean = 0
stddev = 100
noise_gauss = np.zeros(image_gray.shape, np.uint8)
cv2.randn(noise_gauss, mean, stddev)
image_noise_gauss = cv2.add(image_gray, noise_gauss)

# Импульсный шум (соль-перец)
noise = np.random.randint(0, 101, size=(image_gray.shape[0], image_gray.shape[1]), dtype=int)
zeros_pixel = np.where(noise == 0)
ones_pixel = np.where(noise == 100)
image_sp = image_gray.copy()
image_sp[zeros_pixel] = 0
image_sp[ones_pixel] = 255

# 2. Протестировать фильтры с различными параметрами
print("2. Тестирование фильтров:")

results = {}

# Медианный фильтр - разные размеры ядра
for ksize in [3, 5, 7]:
    filtered = cv2.medianBlur(image_noise_gauss, ksize)
    mse = mean_squared_error(image_gray, filtered)
    ssim = structural_similarity(image_gray, filtered)
    results[f'median_{ksize}_gauss'] = (mse, ssim)
    print(f"Медианный {ksize}x{ksize}: MSE={mse:.1f}, SSIM={ssim:.4f}")

# Гауссов фильтр - разные параметры
for ksize in [3, 5]:
    for sigma in [1.0, 2.0]:
        filtered = cv2.GaussianBlur(image_noise_gauss, (ksize, ksize), sigma)
        mse = mean_squared_error(image_gray, filtered)
        ssim = structural_similarity(image_gray, filtered)
        results[f'gauss_{ksize}_{sigma}_gauss'] = (mse, ssim)
        print(f"Гауссов {ksize}x{ksize}, sigma={sigma}: MSE={mse:.1f}, SSIM={ssim:.4f}")

# Билатеральный фильтр
filtered = cv2.bilateralFilter(image_noise_gauss, 9, 75, 75)
mse = mean_squared_error(image_gray, filtered)
ssim = structural_similarity(image_gray, filtered)
results['bilateral_gauss'] = (mse, ssim)
print(f"Билатеральный: MSE={mse:.1f}, SSIM={ssim:.4f}")

# Нелокальные средние
filtered = cv2.fastNlMeansDenoising(image_noise_gauss, h=20)
mse = mean_squared_error(image_gray, filtered)
ssim = structural_similarity(image_gray, filtered)
results['nlm_gauss'] = (mse, ssim)
print(f"Нелокальные средние: MSE={mse:.1f}, SSIM={ssim:.4f}")

# 3. Выяснить, какой фильтр показал лучший результат
print("3. Лучший фильтр для гауссова шума:")

best_mse = min(results.items(), key=lambda x: x[1][0])
best_ssim = max(results.items(), key=lambda x: x[1][1])

print(f"По MSE: {best_mse[0]} - {best_mse[1][0]:.1f}")
print(f"По SSIM: {best_ssim[0]} - {best_ssim[1][1]:.4f}")

# Простая визуализация результатов
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('Исходное')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(image_noise_gauss, cmap='gray')
plt.title('Гауссов шум')
plt.axis('off')

# Покажем лучший по SSIM фильтр
if 'median' in best_ssim[0]:
    ksize = int(best_ssim[0].split('_')[1])
    best_result = cv2.medianBlur(image_noise_gauss, ksize)
elif 'gauss' in best_ssim[0]:
    ksize = int(best_ssim[0].split('_')[1])
    sigma = float(best_ssim[0].split('_')[2])
    best_result = cv2.GaussianBlur(image_noise_gauss, (ksize, ksize), sigma)
elif 'bilateral' in best_ssim[0]:
    best_result = cv2.bilateralFilter(image_noise_gauss, 9, 75, 75)
else:
    best_result = cv2.fastNlMeansDenoising(image_noise_gauss, h=20)

plt.subplot(1, 4, 3)
plt.imshow(best_result, cmap='gray')
plt.title(f'Лучший: {best_ssim[0]}')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(image_sp, cmap='gray')
plt.title('Импульсный шум')
plt.axis('off')

plt.tight_layout()
plt.show()
