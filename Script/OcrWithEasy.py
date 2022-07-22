import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np


def elim_dupl(string):
	p=""
	for char in string:
		if char not in p:
			p=p+char
	return p

def calcola_errore(text,img_nome):
	text = elim_dupl(text)
	img_nome = elim_dupl(img_nome)
	massimo = max(len(text),len(img_nome))
	if massimo == len(text)-1:
		massimo = text
		minimo = img_nome
	else: 
		massimo = img_nome
		minimo = text
	count = 0
	for a in range(len(minimo)-1):
		for b in range(len(massimo)-1):
			if minimo[a] == massimo[b]:
				count = count + 1
	if len(massimo) == 0:
		return str(0)
	else:
		stringa_percentuale = str((100*count)/len(massimo))
		return stringa_percentuale[0:3] +"%"

img_nome = 'CV512WK.jpg'

img = cv2.imread(img_nome)

reader = easyocr.Reader(['en'])
result = reader.readtext(img_nome)
result
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread(img_nome)
spacer = 100

for detection in result: 
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    text = detection[1]
    text  = text + 'error: '+calcola_errore(text,img_nome)
    img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
    img = cv2.putText(img,text,(20,spacer), font, 0.5,(0,255,0),2,cv2.LINE_AA)
    cv2.imwrite('predicted.jpg',img)
    spacer+=15
    
plt.imshow(img)
plt.show()
