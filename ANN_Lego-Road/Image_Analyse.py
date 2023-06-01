import cv2
import matplotlib.pyplot as plt


def Gray_Binary(file_path, shape, thresh):
    img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img_gray, shape, interpolation=cv2.INTER_AREA)
    img_binary = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_resize, img_binary

plt.figure()
for f in range(1, 21):
    file = "../File_Lego-Road/Forward/" + str(f) + ".jpg"
    binary = Gray_Binary(file, (100, 100), 180)[1]
    plt.subplot(5, 4, f)
    plt.title(str(f))
    plt.imshow(binary, cmap='gray')

plt.figure()
for l in range(1, 21):
    file = "../File_Lego-Road/Left/" + str(l) + ".jpg"
    binary = Gray_Binary(file, (100, 100), 180)[1]
    plt.subplot(5, 4, l)
    plt.title(str(l))
    plt.imshow(binary, cmap='gray')

plt.figure()
for r in range(1, 21):
    file = "../File_Lego-Road/Right/" + str(r) + ".jpg"
    binary = Gray_Binary(file, (100, 100), 180)[1]
    plt.subplot(5, 4, r)
    plt.title(str(r))
    plt.imshow(binary, cmap='gray')

plt.show()



