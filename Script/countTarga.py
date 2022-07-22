import os,shutil

#os.chdir('foto')
print(os.getcwd())
lista_file = os.listdir(os.getcwd())
cout = 0
for file in lista_file:
	if '.txt' not in file:
		continue
	else:
		lista_righe = []
		reader = open(file,'r')
		for riga in reader.readlines():
			if len(riga.split()) > 0 and riga.split()[0] == '0':
				lista_righe.append(riga) 
		reader.close()
		count = count + len(lista_righe)
print("Totale targhe: ", count)

