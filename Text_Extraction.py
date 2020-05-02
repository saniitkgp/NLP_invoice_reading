import cv2
import os
import sys
from pdf2jpg import pdf2jpg
import pytesseract
import pickle
from pathlib import Path
import shutil
import sys
TESERRACT_PATH = 'C:/Program Files/Tesseract-OCR/tesseract'
TESSERACT_LANG = 'eng'
TESSERACT_PSM_MODE = '--psm 6'
inp_path = str(sys.argv[1])

def PdftoJpg(inp_path):
    inp_path = Path(inp_path)
    out_path = str(inp_path)+'/tmp'
    os.makedirs(out_path+'/text', exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    filenames = [x for x in os.listdir(inp_path) if x.endswith('.pdf')]
    x = [pdf2jpg.convert_pdf2jpg(str(inp_path)+'/'+filename, str(out_path)+'/'+filename, pages="ALL") for filename in filenames]
    y = [[shutil.move(str(out_path)+'/'+filename+'/'+filename+'_dir'+'/'+ file,str(out_path)+'/'+filename+'/'+ file) for file in os.listdir(str(out_path)+'/'+filename+'/'+filename+'_dir')] for filename in filenames]
#     inp2 = [str(out_path)+'/'+filename+'/' for filename in filenames]
#     for inp in inp2:
#         imgfile = [str(inp)+'/'+x for x in os.listdir(inp) if x.endswith('.jpg')]
#         img = [cv2.imread(x) for x in imgfile]
#         gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in img]
#         for i in range(len(gray)):
#             text = pytesseract.image_to_string(gray[i], lang=TESSERACT_LANG, config=TESSERACT_PSM_MODE)
#             with open(str(imgfile[i])+'.txt', 'w') as f:
#                 f.write(text)
#             shutil.move(str(imgfile[i])+'.txt', out_path+'/text/')

PdftoJpg(inp_path)