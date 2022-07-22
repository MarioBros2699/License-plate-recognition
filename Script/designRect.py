import cv2 as cv
import matplotlib.pyplot as plt

left_x, top_y, w, h = [312, 447, 73, 29]
img = cv.imread('JIL8091.jpg')

#inverso di roi
end_point = (left_x,top_y)
start_point = (left_x+w,top_y+h)

#baricentro
x_g = int(left_x+w/2)
y_g = int((top_y+h)-h/2)
g_point = (x_g,y_g)


image = cv.circle(img, g_point, radius=0, color=(0, 0, 255), thickness=-1)
img = cv.rectangle(img, start_point, end_point, color=(0, 0, 255), thickness = 2)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()