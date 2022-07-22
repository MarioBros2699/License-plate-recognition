import os,shutil
from PIL import Image

def filtra(lista_file):
	for file in lista_file:
		if len(file) < 20 or 'DEL' in file:
			lista_file.remove(file) 
	return lista_file

def pulisciCartella():
	lista = os.listdir(os.getcwd())
	for file in lista:
		if 'DEL' in file:
			os.remove(file)


#os.chdir('foto')
print(os.getcwd())
lista_file = filtra(os.listdir(os.getcwd()))
i = int(len(lista_file)/2) - 1
vecchia = int(len(lista_file)/2)
for file_1 in lista_file:
	if '.jpg' in file_1:
		for file_2 in lista_file:
			if file_1.replace(".jpg", ".txt") == file_2:
				im = Image.open(file_1) 
				im.show(title=file_1)
				nuovo_nome = input('Inserisci targa'+'(mancano:'+str(i)+'): ')
				txt = nuovo_nome+'.txt'
				img = nuovo_nome+'.jpg'
				os.rename(file_2,txt)
				os.rename(file_1,img)
				os.system("TASKKILL /F /IM Microsoft.Photos.exe")
				pulisciCartella()
				lista_file = filtra(os.listdir(os.getcwd()))
				i = vecchia - int(len(lista_file)/2)
				vecchia = i
				
	else:
		continue	
		