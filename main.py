# Импортируем зависимости
import matplotlib.pyplot as plt

import cv2
import pytesseract

# Задаём путь Tesseract CMD до файла tesseract.exe, который мы установили заранее
pytesseract.pytesseract.tesseract_cmd = r''  # 'C:/path/to/you/tesseract_directory

# Читаем изображение и конвертируем в RGB
carplate_img = cv2.imread('./images/car_image3.jpg')  # Необходимо создать и поместить изображения в директорию /images
carplate_img_rgb = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)

plt.imshow(carplate_img_rgb)


# Функция для приведения изображения к нужному размеру
def enlarge_plt_display(image, scale_factor):
    width = int(image.shape[1] * scale_factor / 100)
    height = int(image.shape[0] * scale_factor / 100)
    dim = (width, height)
    plt.figure(figsize=dim)
    plt.axis('off')
    plt.imshow(image)


enlarge_plt_display(carplate_img_rgb, 1.2)

# Импортируем Haar Cascade XML для Российских номеров
carplate_haar_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_russian_plate_number.xml')


# Настраиваем функцию для поиска автомобильных номеров
def carplate_detect(image):
    carplate_overlay = image.copy()
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=3)

    for x, y, w, h in carplate_rects:
        cv2.rectangle(carplate_overlay, (x, y), (x + w, y + h), (255, 0, 0), 5)

        return carplate_overlay


detected_carplate_img = carplate_detect(carplate_img_rgb)
enlarge_plt_display(detected_carplate_img, 1.2)


# Функция для извлечения номеров
def carplate_extract(image):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y + 15:y + h - 10, x + 15:x + w - 20]

    return carplate_img


# Приведение изображения номерного знака к нужному размеру
def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


# Выводим изображение ГРН на экран
carplate_extract_img = carplate_extract(carplate_img_rgb)
carplate_extract_img = enlarge_img(carplate_extract_img, 150)
plt.imshow(carplate_extract_img)

# Конвертируем изображение в оттенках серого
carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)
plt.axis('off')
plt.imshow(carplate_extract_img_gray, cmap='gray')

# Применяем среднее размытие и изображение в оттенках серого
carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray, 3)  # Kernel size 3
plt.axis('off')
plt.imshow(carplate_extract_img_gray_blur, cmap='gray')

# Выводим в консоль распознанный номер
print(pytesseract.image_to_string(carplate_extract_img_gray_blur,
                                  config=f'--psm 8 --oem 3 -c '
                                         f'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

# Тестируем все PSM значения
for i in range(3, 14):
    print(f'PSM: {i}')
    print(pytesseract.image_to_string(carplate_extract_img_gray_blur,
                                      config=f'--psm {i} --oem 3 -c tessedit_char_whitelist'
                                             f'=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
