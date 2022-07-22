import cv2
import numpy as np
from matplotlib import pyplot as plt

def basic_oper(image,iter_dilat,iter_erode):
	kernel = np.ones((3,3))
	imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
	imgCanny = cv2.Canny(imgBlur,150,200)
	imgDial = cv2.dilate(imgCanny,kernel,iterations=iter_dilat)
	cv2.imwrite('dilat.jpg',imgDial)
	imgThres = cv2.erode(imgDial,kernel,iterations=iter_erode)
	cv2.imwrite('erod.jpg',imgThres)
	return imgGray, imgBlur, imgCanny, imgDial, imgThres

def get_contours(img, orig): 
    maxContour = np.array([])
    maxArea = 0
    imgContour = orig.copy()  
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                maxContour = approx
                maxArea = area
                index = i
    warped = None  
    if index is not None: 
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)
        punti_contorno = np.squeeze(maxContour).astype(np.float32)
        height = image.shape[0]
        width = image.shape[1]
        dst = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]])
        M = cv2.getPerspectiveTransform(punti_contorno, dst)
        img_shape = (width, height)
        warped = cv2.warpPerspective(orig, M, img_shape, flags=cv2.INTER_LINEAR)
    return maxContour, imgContour, warped , dst

def get_trasf(image, warped, dst, h_t, scale, h_s, rotaz, h_r):
	#affine
	height = image.shape[0]
	width = image.shape[1]
	rows, cols, ch = warped.shape
	src2 = dst[0:len(dst)-1]
	src2[len(src2)-1] = dst[len(dst)-1]
	dst2 = np.float32([[0, h_t], [scale, h_s], [rotaz, h_r]])
	M2 =cv2.getAffineTransform(src2,dst2)
	trasf = cv2.warpAffine(warped, M2, (700,300))
	return trasf

img_targa = 'CY320BR_1.jpg'
image = cv2.imread(img_targa)
iter_dilat = int(input('iserisci valore di dilate: '))
iter_erode = int(input('iserisci valore di erode: '))

imgGray, imgBlur, imgCanny, imgDial, imgThres = basic_oper(image,iter_dilat,iter_erode)
try:
	maxContour, imgContour, warped, dst = get_contours(imgThres, image)
except UnboundLocalError:
	print('riprova con altri parametri. Guarda dilat e erod.jpg per capire meglio')
	exit()

cv2.imwrite('warp.jpg',warped)
print("Wrap fatto, imposta fattori per la trasformazione affine(vedi warp.png)")
h_t = int(input('iserisci traslazione-h (0-300): '))
scale = int(input('iserisci larghezza per il punto di scala (0:700): '))
h_s = int(input('iserisci scala-h (0-300): '))
rotaz = int(input('iserisci larghezza per il punto di rotazione (0:700): '))
h_r = int(input('iserisci rotazione-h (0-300): '))
trasf = get_trasf(image, warped, dst, h_t, scale, h_s, rotaz, h_r)
cal_nome = img_targa[0:len(img_targa)-6]+'.jpg'
cv2.imwrite(cal_nome,trasf)

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
trasf = cv2.cvtColor(trasf, cv2.COLOR_BGR2RGB)

plt.subplot(4, 2, 1)
plt.imshow(imgGray , cmap='gray')
plt.title('Gray')

plt.subplot(4, 2, 2)
plt.imshow(imgBlur ,cmap='gray')
plt.title('Blur')

plt.subplot(4, 2, 3)
plt.imshow(imgCanny, cmap='gray')
plt.title('edge detect')

plt.subplot(4, 2, 4)
plt.imshow(imgDial, cmap='gray')
plt.title('dilatazione')

plt.subplot(4, 2, 5)
plt.imshow(imgThres, cmap='gray')
plt.title('erode')

plt.subplot(4, 2, 6)
plt.imshow(imgContour)
plt.title('contorno massimo')

plt.subplot(4, 2, 7)
plt.imshow(warped)
plt.title('warp persp.')

plt.subplot(4, 2, 8)
plt.imshow(trasf)
plt.title('affine trasf.')

plt.subplots_adjust(hspace = 1)
plt.show()