import os,shutil
import numpy as np

# Current directory
dir_partenza = os.getcwd()
print ('Parti da: ',os.getcwd())

#passo a data e micreo i .txt necessari
os.chdir('data')
file_train = open('train.txt', 'w')
file_valid = open('valid.txt', 'w')
file_test = open('test.txt', 'w')

#passo alla cartella obj
os.chdir('obj')
print('Sei nella cartella: ',os.getcwd())
lista_file_obj = os.listdir(os.getcwd())
filtrata = []
for file_obj in lista_file_obj:
	if '.txt' not in file_obj and '.ini' not in file_obj:
		filtrata.append('data/obj/'+file_obj)
print('Ho trovato: ',len(filtrata), 'immagini')

#scrivo all'interno dei file le immagini mischiate 
np.random.shuffle(filtrata)
perc_train = 75 #75%
perc_val_test = 12.5 #12.5%
tot_img = len(filtrata)
tot_train = int((tot_img*perc_train)/100)
tot_valtest = int((tot_img*perc_val_test)/100)
print(tot_train,tot_valtest)
count = 0
for file in filtrata:
	if count < tot_train:
		file_train.write(file+'\n')
	elif count <= tot_train+tot_valtest and count > tot_train:
		file_valid.write(file+'\n')
	else:
		file_test.write(file+'\n')
	count = count + 1

#chiudo i flussi ai file
file_train.close()
file_valid.close()
file_test.close()