import numpy as np
from numpy import loadtxt,array
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_curve,auc,precision_score,recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from time import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages

if __name__== "__main__":
    startTime = time()

    train_data = fetch_20newsgroups(
        subset='train',
        shuffle=True,
        random_state=42,
    )
    test_data = fetch_20newsgroups(
        subset='test',
        shuffle=True,
        random_state=42,
    )
    Xtrain = train_data.data
    Ytrain = train_data.target
    Xtest = test_data.data
    Ytest = test_data.target
    namelist = train_data.target_names

    ### Text processing
    cv = CountVectorizer()
    Xtrain_c = cv.fit_transform(Xtrain)
    Xtest_c = cv.transform(Xtest)
    tf = TfidfTransformer(use_idf=False).fit(Xtrain_c)
    Xtrain_tf = tf.transform(Xtrain_c)
    Xtest_tf = tf.transform(Xtest_c)
    del Xtrain,Xtest,Xtrain_c, Xtest_c

    
    ## Learning
    C = 1
    svm = SVC(C=C, kernel=cosine_similarity,probability=True)
    nb = MultinomialNB()
    svmTime = time()
    svm.fit(Xtrain_tf,Ytrain)
    svmTime = time()-svmTime
    nbTime = time()
    nb.fit(Xtrain_tf,Ytrain)
    nbTime = time()-nbTime
    svm_pred = svm.predict(Xtest_tf)
    svm_pred_prob = svm.predict_proba(Xtest_tf)
    nb_pred = nb.predict(Xtest_tf)
    nb_pred_prob = nb.predict_proba(Xtest_tf)
    nb_acc =accuracy_score (Ytest,nb_pred)
    svm_acc = accuracy_score(Ytest,svm_pred)
    nb_precise = precision_score(Ytest,nb_pred,average='micro')
    svm_precise = precision_score(Ytest,svm_pred,average='micro')
    nb_recall = recall_score(Ytest,nb_pred,average='micro')
    svm_recall = recall_score(Ytest,svm_pred,average='micro')
    '''print("SVM accuracy Score: ", accuracy_score(Ytest,svm_pred))

    print("NB accuracy Score: ", accuracy_score(Ytest,nb_pred))
    print ("Elapsed time:", time()-startTime)'''
    print(tabulate([['SVM',svm_acc,svm_precise,svm_recall,svmTime],['Naive Bayes',nb_acc,nb_precise,nb_recall,nbTime]], headers=[ 'algorimth', 'accuracy','precision','recall','train time']))
    
    ### Plot.
    target_cat = ['comp.graphics', 'comp.sys.mac.hardware', 'rec.motorcycles', 'sci.space', 'talk.politics.mideast']
    cat_id = []
   
    fig= []
    with PdfPages("Questions\CIS419-master\Assignment4\hw4_skeleton_20171106\graphTextClassifierROC.pdf") as pp:
        for t in target_cat:
            id = namelist.index(t)
            
            #print (cat_id)
            

            c1 = nb_pred_prob[:,id]
            fpr,tpr,thresold = roc_curve(Ytest,c1,pos_label=id)
            roc_auc = auc(fpr,tpr)
            
            

           
            plt.title('ROC Naive bayes, class: '+ str(t))
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            pp.savefig()
            plt.close()
            

            #plt.show()
           
            
            c1 = svm_pred_prob[:,id]
            fpr,tpr,thresold = roc_curve(Ytest,c1,pos_label=id)
            roc_auc = auc(fpr,tpr)


           
            plt.title('ROC Support Vector Machine, class: '+ str(t))
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            pp.savefig()
            plt.close()
            
        
    
        




