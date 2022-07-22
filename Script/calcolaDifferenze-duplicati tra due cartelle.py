import os,shutil

cartella1 = input("Inserisci cartella 1: ")
cartella2 = input("Inserisci cartella 2-elimino qui-: ")

# Current directory
dir_partenza = os.getcwd()
print ('Parti da: ',os.getcwd())

#passo a obj
os.chdir(cartella1)
print('Sei nella cartella: ',os.getcwd())
lista_file_obj = os.listdir(os.getcwd())
print('Ho trovato: ',len(lista_file_obj))

#cambio a ImgValidation
os.chdir(dir_partenza+'\\'+cartella2)
print('Sei nella cartella: ',os.getcwd())
lista_file_val = os.listdir(os.getcwd())
print('Ho trovato: ', len(lista_file_val))

#confronto le liste e calcolo la lista differenza: 
lista_uguali = []
for file_val in lista_file_val:
	for file_obj in lista_file_obj:
		if (file_val == file_obj) and ('.ini' not in file_val or '.ini' not in file_obj):
			lista_uguali.append(file_val)
print('Ho trovato ',len(lista_uguali), ' file uguali')

#stampo la differenza in un txt
#os.chdir(dir_partenza)
#with open('file_diff.txt', 'w') as writer:
#    for file in lista_uguali:
#    	writer.write(file+'\n')

#elimino i duplicati in ImgValidation
for file in lista_uguali:
	os.remove(file)

