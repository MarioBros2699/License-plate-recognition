from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib import cm
import pytesseract
from skimage import feature,filters
import numpy as np
import cv2
import os


def get_cordinate(riga_targa):
	parte_confi = riga_targa.split('(')[0]
	confidenza = int((parte_confi.split(':')[1].split('%')[0]))/100
	parte_cordinate = riga_targa.split('(')[1].split(')')[0]
	left_x = int(parte_cordinate.split('left_x:')[1][0:7])
	top_y = int(parte_cordinate.split('top_y:')[1][0:7])
	w = int(parte_cordinate.split('width:')[1][0:7])
	h = int(parte_cordinate.split('height:')[1][0:7])
	return left_x,top_y,w,h

def set_roi(riga_targa, nome_img, img, count, path):
	left_x, top_y, w, h = get_cordinate(riga_targa)
	roi = (left_x,top_y,left_x+w,top_y+h)
	targa = img.crop(roi).resize((500 , 218))  
	targa.save(path+nome_img[0:len(nome_img)-4]+'_'+str(count)+'.jpg')

risultati_txt = open('risultati.txt', 'r')
righe_saltare = 13
lista_riga = risultati_txt.readlines()
for i in range(len(lista_riga)-1):
	line = lista_riga[i]
	if 'data/obj/'in line:
		nome_img = line.replace('data/obj/','').split(':')[0]
		print(nome_img)
		try:
			img = Image.open(os.getcwd()+'/Test/'+nome_img)
			path = os.getcwd()+'/Roi/'
			count = 1
			indice = i+1
			riga_targa = lista_riga[indice]
			while 'targa' in riga_targa:
				set_roi(riga_targa, nome_img, img, count, path)
				count = count + 1
				indice = indice + 1
				riga_targa = lista_riga[indice]
		except FileNotFoundError:
			print('Non ho trovato un file')