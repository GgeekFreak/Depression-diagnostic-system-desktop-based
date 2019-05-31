import pandas as pd
import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
import nltk
import gensim
from nltk.corpus import stopwords
from MeanWord2vec import MeanEmbeddingVectorizer
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication,QMessageBox
import sys
from mainui1 import Ui_Form

data = pd.read_csv('database 1- Copy.csv',skiprows=1,header=None,engine='python')
df = pd.DataFrame(data)
X = df.iloc[0:,1:-1]
Y = df.iloc[0:,-1]

features = []
stop = []
testfeatures = []
Questions = []

f = open('survey_14_questions.txt','r')
for lines in f:
    Questions.append(lines)

print(Questions)
Answers = []

class mainUi(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        global Answers
        QtWidgets.QWidget.__init__(self)
        self.InutUi()
        Answers = ["" for i in range(14)]

    def InutUi(self):
        self.cur_index = 0
        self.setupUi(self)
        self.show()
        self.flag = 0
        self.index = 0
        self.counter = 0
        self.btn.clicked.connect(self.clickstate)
        self.Backbtn.clicked.connect(self.backclick)
        self.predict_label.setVisible(0)
        self.face_smile_label.setVisible(0)
        self.face_sad_label.setVisible(0)
        self.l.setText(Questions[self.cur_index])
        self.counter_label.setText('1 out of 14')
        self.Backbtn.setDisabled(1)
    def backclick(self):
        if(self.index > 0):
            self.index -= 1
            self.counter -= 1
            self.counter_label.setText('{} out of 14'.format(self.index + 1))
            self.l.setText(Questions[self.index])
            self.lineEdit.setText(Answers[self.index])
            self.cur_index-=1
        else:
            self.Backbtn.setDisabled(1)

    def refreshText(self):
        self.lineEdit.setText('')

    def clickstate(self):
       self.updateText()
       self.lineEdit.setText(Answers[self.cur_index])
    def updateText(self):


           self.cur_answer = self.lineEdit.text()
           if(len(self.cur_answer) > 0):
              self.Backbtn.setDisabled(0)
              self.index += 1
              if(self.index < 13):

                  self.counter_label.setText('{} out of 14'.format(self.index + 1))
                  self.cur_answer = self.lineEdit.text()
                  Answers[self.cur_index] = self.cur_answer
                  #Answers.insert(self.index,self.cur_answer)
                  self.l.setText(Questions[self.index])
                  self.refreshText()
                  self.cur_index += 1
                  print(self.index)
                  print(Answers)

              elif(self.index == 13):
                  self.counter_label.setText('{} out of 14'.format(self.index + 1))
                  self.cur_answer = self.lineEdit.text()
                  Answers[self.cur_index] = self.cur_answer
                  self.l.setText(Questions[self.index])
                  self.btn.setText('Show Result')
                  print(Answers)
                  self.flag = 1
                  self.refreshText()
                  self.cur_index += 1
              elif(self.flag > 0):

                      self.cur_answer = self.lineEdit.text()

                      Answers[self.cur_index] = self.cur_answer
                      self.refreshText()
                      print(Answers)
                      self.Trainer('GoogleNews-vectors-negative300.bin')
                      self.btn.setDisabled(1)
                      self.Backbtn.setDisabled(1)
                      self.lineEdit.setVisible(0)
                      self.l.setVisible(0)
                      self.flag = 0
           else:
               msg = QMessageBox(self)
               msg.setText('Incorrect answer, please try again !! ')
               msg.exec()

    def Trainer(self, model_name):
        testdf = pd.DataFrame(np.array(Answers).reshape(1, 14))

        for i in stopwords.words('english'):

            if (not i == "no" and not i == "all" and not i == 'not'):
                stop.append(i)


        model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True,
                                                 limit=200000)
        for cols in X:
            test_tokens = [nltk.word_tokenize(re.sub(r'[^\w\s]', '', sent).lower()) for sent in X[cols]]
            filtered_tokens = [[w for w in sent if w not in
                                stop and w != "n't"] for sent in test_tokens]


            meanvects = MeanEmbeddingVectorizer(model)
            words = meanvects.fit_transform(filtered_tokens)

            features.append(words)

        xtrain = np.concatenate(features, axis=1)

        for i in testdf:
            tokens = [nltk.word_tokenize(re.sub(r'[^\w\s]', '', sent).lower()) for sent in testdf[i]]
            test_filtered_tokens = [[w for w in sent if w not in
                                     stop and w != "n't"] for sent in tokens]

            meanembedding = MeanEmbeddingVectorizer(model)
            vec = meanembedding.fit_transform(test_filtered_tokens)

            testfeatures.append(vec)

        test_features = np.array(testfeatures)

        xtest = np.concatenate(test_features, axis=1)

        clf = GaussianNB()
        m = clf.fit(xtrain,Y)
        self.preds = clf.predict(xtest)

        if self.preds == 0:
            self.predict_label.setText('Congrats .. you are not depressed')
            self.predict_label.setVisible(1)
            self.face_smile_label.setVisible(1)
        else:
            self.predict_label.setText('ufortunately,it seems you are despressed')
            self.predict_label.setVisible(1)
            self.face_sad_label.setVisible(1)

def main():
    args = list(sys.argv)
    args[1:1] = ['-stylesheet','aqua.qss']
    app = QApplication(args)
    ui = mainUi()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
