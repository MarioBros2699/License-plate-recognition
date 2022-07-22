from PIL import Image
from pathlib import Path
import os

def get_cordinate(riga_targa):
	parte_confi = riga_targa.split('(')[0]
	confidenza = int((parte_confi.split(':')[1].split('%')[0]))/100
	parte_cordinate = riga_targa.split('(')[1].split(')')[0]
	left_x = int(parte_cordinate.split('left_x:')[1][0:7])
	top_y = int(parte_cordinate.split('top_y:')[1][0:7])
	w = int(parte_cordinate.split('width:')[1][0:7])
	h = int(parte_cordinate.split('height:')[1][0:7])
	print(confidenza,left_x,top_y,w,h) 
	x_g_perc = (left_x+w/2)/(w_image)
	y_g_perc = ((top_y+h)-h/2)/(h_image)
	w_perc = w/w_image
	h_perc = h/h_image
	return "0"+" "+str(confidenza)+' '+str(x_g_perc)+" "+str(y_g_perc)+" "+str(w_perc)+" "+str(h_perc)+"\n"


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
			w_image,h_image = img.size
			nome_label = nome_img[0:len(nome_img)-4]+'.txt'
			label_txt = open(os.getcwd()+'/Label/'+nome_label,'w')
			indice = i+1
			riga_targa = lista_riga[indice]
			while 'targa' in riga_targa:
				cordinate_label= get_cordinate(riga_targa)
				label_txt.write(cordinate_label)
				indice = indice + 1
				riga_targa = lista_riga[indice]
			label_txt.close()
		except FileNotFoundError:
			print('Non ho trovato un file')