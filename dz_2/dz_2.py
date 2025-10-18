import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error

image = cv2.imread('sar_1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


mean = 0
stddev = 100
noise_gauss = np.zeros(image_gray.shape, np.uint8)
cv2.randn(noise_gauss, mean, stddev)
image_noise_gauss = cv2.add(image_gray, noise_gauss)

noise = np.random.randint(0, 101, size=(image_gray.shape[0], image_gray.shape[1]), dtype=int)
zeros_pixel = np.where(noise == 0)
ones_pixel = np.where(noise == 100)
image_sp = image_gray.copy()
image_sp[zeros_pixel] = 0
image_sp[ones_pixel] = 255

print("2. Тестирование фильтров для ГАУССОВА шума:")
results_gauss = {}

for ksize in [3, 5, 7]:
    filtered = cv2.medianBlur(image_noise_gauss, ksize)
    mse = mean_squared_error(image_gray, filtered)
    ssim = structural_similarity(image_gray, filtered)
    results_gauss[f'median_{ksize}'] = (mse, ssim)
    print(f"Медианный {ksize}x{ksize}: MSE={mse:.1f}, SSIM={ssim:.4f}")

for ksize in [3, 5]:
    for sigma in [1.0, 2.0]:
        filtered = cv2.GaussianBlur(image_noise_gauss, (ksize, ksize), sigma)
        mse = mean_squared_error(image_gray, filtered)
        ssim = structural_similarity(image_gray, filtered)
        results_gauss[f'gauss_{ksize}_{sigma}'] = (mse, ssim)
        print(f"Гауссов {ksize}x{ksize}, sigma={sigma}: MSE={mse:.1f}, SSIM={ssim:.4f}")

filtered = cv2.bilateralFilter(image_noise_gauss, 9, 75, 75)
mse = mean_squared_error(image_gray, filtered)
ssim = structural_similarity(image_gray, filtered)
results_gauss['bilateral'] = (mse, ssim)
print(f"Билатеральный: MSE={mse:.1f}, SSIM={ssim:.4f}")

filtered = cv2.fastNlMeansDenoising(image_noise_gauss, h=20)
mse = mean_squared_error(image_gray, filtered)
ssim = structural_similarity(image_gray, filtered)
results_gauss['nlm'] = (mse, ssim)
print(f"Нелокальные средние: MSE={mse:.1f}, SSIM={ssim:.4f}")

print("\n3. Тестирование фильтров для ИМПУЛЬСНОГО шума:")
results_sp = {}

for ksize in [3, 5, 7]:
    filtered = cv2.medianBlur(image_sp, ksize)
    mse = mean_squared_error(image_gray, filtered)
    ssim = structural_similarity(image_gray, filtered)
    results_sp[f'median_{ksize}'] = (mse, ssim)
    print(f"Медианный {ksize}x{ksize}: MSE={mse:.1f}, SSIM={ssim:.4f}")

for ksize in [3, 5]:
    for sigma in [1.0, 2.0]:
        filtered = cv2.GaussianBlur(image_sp, (ksize, ksize), sigma)
        mse = mean_squared_error(image_gray, filtered)
        ssim = structural_similarity(image_gray, filtered)
        results_sp[f'gauss_{ksize}_{sigma}'] = (mse, ssim)
        print(f"Гауссов {ksize}x{ksize}, sigma={sigma}: MSE={mse:.1f}, SSIM={ssim:.4f}")

filtered = cv2.bilateralFilter(image_sp, 9, 75, 75)
mse = mean_squared_error(image_gray, filtered)
ssim = structural_similarity(image_gray, filtered)
results_sp['bilateral'] = (mse, ssim)
print(f"Билатеральный: MSE={mse:.1f}, SSIM={ssim:.4f}")

filtered = cv2.fastNlMeansDenoising(image_sp, h=20)
mse = mean_squared_error(image_gray, filtered)
ssim = structural_similarity(image_gray, filtered)
results_sp['nlm'] = (mse, ssim)
print(f"Нелокальные средние: MSE={mse:.1f}, SSIM={ssim:.4f}")

# 4. Выяснить лучшие фильтры для каждого типа шума

best_mse_gauss = min(results_gauss.items(), key=lambda x: x[1][0])
best_ssim_gauss = max(results_gauss.items(), key=lambda x: x[1][1])

print(f"ДЛЯ ГАУССОВА ШУМА:")
print(f"  По MSE:  {best_mse_gauss[0]} - {best_mse_gauss[1][0]:.1f}")
print(f"  По SSIM: {best_ssim_gauss[0]} - {best_ssim_gauss[1][1]:.4f}")

best_mse_sp = min(results_sp.items(), key=lambda x: x[1][0])
best_ssim_sp = max(results_sp.items(), key=lambda x: x[1][1])

print(f"\nДЛЯ ИМПУЛЬСНОГО ШУМА:")
print(f"  По MSE:  {best_mse_sp[0]} - {best_mse_sp[1][0]:.1f}")
print(f"  По SSIM: {best_ssim_sp[0]} - {best_ssim_sp[1][1]:.4f}")




plt.figure(figsize=(15, 5))

plt.subplot(1, 5, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('Исходное')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(image_noise_gauss, cmap='gray')
plt.title('Гауссов шум')
plt.axis('off')


if 'median' in best_ssim_gauss[0]:
    ksize = int(best_ssim_gauss[0].split('_')[1])
    best_gauss_result = cv2.medianBlur(image_noise_gauss, ksize)
elif 'gauss' in best_ssim_gauss[0]:
    ksize = int(best_ssim_gauss[0].split('_')[1])
    sigma = float(best_ssim_gauss[0].split('_')[2])
    best_gauss_result = cv2.GaussianBlur(image_noise_gauss, (ksize, ksize), sigma)
elif 'bilateral' in best_ssim_gauss[0]:
    best_gauss_result = cv2.bilateralFilter(image_noise_gauss, 9, 75, 75)
else:
    best_gauss_result = cv2.fastNlMeansDenoising(image_noise_gauss, h=20)

plt.subplot(1, 5, 3)
plt.imshow(best_gauss_result, cmap='gray')
plt.title(f'Лучший для гауссова\n{best_ssim_gauss[0]}')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(image_sp, cmap='gray')
plt.title('Импульсный шум')
plt.axis('off')


if 'median' in best_ssim_sp[0]:
    ksize = int(best_ssim_sp[0].split('_')[1])
    best_sp_result = cv2.medianBlur(image_sp, ksize)
elif 'gauss' in best_ssim_sp[0]:
    ksize = int(best_ssim_sp[0].split('_')[1])
    sigma = float(best_ssim_sp[0].split('_')[2])
    best_sp_result = cv2.GaussianBlur(image_sp, (ksize, ksize), sigma)
elif 'bilateral' in best_ssim_sp[0]:
    best_sp_result = cv2.bilateralFilter(image_sp, 9, 75, 75)
else:
    best_sp_result = cv2.fastNlMeansDenoising(image_sp, h=20)

plt.subplot(1, 5, 5)
plt.imshow(best_sp_result, cmap='gray')
plt.title(f'Лучший для импульсного\n{best_ssim_sp[0]}')
plt.axis('off')

plt.tight_layout()
plt.show()