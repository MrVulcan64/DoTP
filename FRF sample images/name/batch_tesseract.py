import os
from subprocess import call

def batchTesseract(folder):
    for filename in os.listdir(folder):
    	if filename.endswith('.png'):
    		call(["tesseract", filename, filename])

def main():
	batchTesseract("/Users/Ardon/Documents/Dropbox/DotP/FRF sample images/name")

if __name__ == '__main__':
	main()