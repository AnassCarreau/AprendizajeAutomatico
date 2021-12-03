import process_email as pmail
import get_vocab_dict as getvoc 
import codecs
import collections
import numpy as np
from sklearn.svm import SVC


def main():
    #leemos  el archivo "vocab.txt" y lo almacenamos como diccionario de pyhon
    #key = palabras , value= orden en diccionario
    vocabDict = collections.OrderedDict(getvoc.getVocabDict())
    X_input = np.zeros(shape=(1000, len(vocabDict))) 
    Y = np.zeros(1000)
    #recorremos los archivos rellenando el vector X segun el spam 
    for i in range(1,500):  
        Y[i] = 1  
        directorio = "spam" 
        email_contents=codecs.open ('{0}/{1:04d}.txt'.format(directorio, i), 'r', encoding = 'utf-8', errors = 'ignore').read()
        email = pmail.email2TokenList(email_contents)
        for j in range (len(email)):
            #si la palabra esta en el diccionario ponemos un 1 en el vector X
            if(email [j] in vocabDict):    
                X_input[i][list(vocabDict.keys()).index(email[j])] = 1
                
    for i in range(1,500): 
        Y[500+i] = 0   
        directorio = "spam" 
        email_contents=codecs.open ('{0}/{1:04d}.txt'.format(directorio, i), 'r', encoding = 'utf-8', errors = 'ignore').read()
        email = pmail.email2TokenList(email_contents)     
        for j in range (len(email)):    
            if(email [j] in vocabDict):    
                X_input[500+i][list(vocabDict.keys()).index(email[j])] = 1 
                
    spam_processor = SVC(C = 1.0, kernel = 'linear')
    spam_processor.fit(X_input, Y.ravel())

main()