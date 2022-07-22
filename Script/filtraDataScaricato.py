import os,shutil
from PIL import Image

os.chdir('foto')
print(os.getcwd())
lista_file = os.listdir(os.getcwd())
for file_1 in lista_file:
	if '.jpg' in file_1:
		for file_2 in lista_file:
			if file_1.replace(".jpg", ".txt") == file_2:
				with Image.open(file_1) as im:
					im.show(title=file_1)
				decision = input('Ha più targhe: ')
				os.system("TASKKILL /F /IM Microsoft.Photos.exe")
				if 'y' not in decision:
					os.remove(file_1)
					os.remove(file_2)
	else:
		continue	
		