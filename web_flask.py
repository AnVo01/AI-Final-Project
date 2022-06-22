from flask import Flask, render_template, request
import keras
import numpy as np
import re
import os
import string
from keras_preprocessing.sequence import pad_sequences
import requests
import pickle
from bs4 import BeautifulSoup

app = Flask(__name__)

dir = 'Viet74K.txt'
file = open(dir,'r',encoding='utf-8')
viet_dict = dict()

for word in file:
  word = re.sub(r'[\n]', '', word)
  if word not in viet_dict:
    viet_dict[word] = len(viet_dict)

def tachtu(text, dict):
  input = text.split(" ")
  words = []
  start = 0
  while True:
    end = len(input)
    while end > start:
      sentence = input[start:end]
      sentence = " ".join(sentence)
      end = end-1 
      if sentence.lower() in viet_dict:
        words.append(sentence)
        break
    start = end + 1  
    if start == len(input):
      break
  output = []
  for word in words:
    word = re.sub(r'[ ]', '_', word)
    output.append(word)
  output = " ".join(output)
  return output

special_character = ['0','1','2','3','4','5','6','7','8','9','!','@','#','$','%','^','&','*','(',')','-','=','+','\n','\t',':',';',',','.','|','"','/','\'']
def xoa_kyhieu(s):
  b = []
  for word in s.split():
    a = []
    for letter in word:
      if letter not in special_character:
        a.append(letter)
    mystring = "".join([str(char) for char in a])
    if mystring != "":
      b.append(mystring)
  mystringfinal = " ".join([str(char) for char in b])
  return mystringfinal

stopword = ['a_lô']
def create_stopword(path):
  with open(path, encoding="utf-8") as words:
    return [w[:len(w) - 1] for w in words] + stopword

stop_words = create_stopword('stop_word.txt')

def xoa_dau(s):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s

def xuly(s):
  s = xoa_kyhieu(s)
  s = tachtu(s,viet_dict)
  s = [word for word in s.lower().split() if word not in stop_words]
  s = " ".join(s)
  s = xoa_dau(s)
  return s

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dictfile = open("word_dict.pkl",'rb')
word_dict = pickle.load(dictfile)
dictfile.close()

model = keras.models.load_model('text_classification_neural.h5')
max_num = 1366
names=['Chính trị Xã hội','Công nghệ','Đời sống','Khoa học','Kinh doanh','Pháp luật','Sức khoẻ','Thế giới','Thể Thao','Văn hoá']


@app.route("/",methods=['POST','GET'])
def home():
  if request.method == "GET":
    return render_template("web.html")
  else:
    text = request.form['text']
    linkurl = request.form['link']
    if len(linkurl) != 0:
      link_res = requests.get(linkurl)
      link_res = str(link_res.text)
      soup = BeautifulSoup(link_res,"html.parser")
      title = soup.title.get_text()
      content = soup.meta['content']
      link_res = title + ' ' + content
      link_res = xuly(link_res).split()
      pred_link = []
      pred_link.append([word_dict[word] for word in link_res if word in word_dict])
      pred_link = pad_sequences(pred_link, max_num)
      result_link = names[np.argmax(model.predict(pred_link))]
      y_pred = list(model.predict(pred_link).reshape(-1,))
      return render_template("web.html",msg = result_link,labels=names,values=y_pred,
                                msg1 = 'Thể loại của văn bản là: ')
      
    if len(text) != 0:
      textxuly = xuly(text).split()
      pred = []
      pred.append([word_dict[word] for word in textxuly if word in word_dict])
      pred = pad_sequences(pred, max_num)
      result = names[np.argmax(model.predict(pred))]
      y_pred = list(model.predict(pred).reshape(-1,))
      return render_template("web.html",msg = result,labels=names,values=y_pred,msg1 = 'Thể loại của văn bản là: ',
                              text1=text)
      
    else:
      y_pred = [0,0,0,0,0,0,0,0,0,0]
      return render_template("web.html",msg1 = 'Vui lòng thêm link HTML hoặc văn bản',labels=names,values=y_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9876,debug=True)
