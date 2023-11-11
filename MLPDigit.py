import numpy as np
from numpy import loadtxt,array
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from time import time
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    startTime = time()
    filename = 'data\digitsX.dat'
    X = loadtxt(filename,delimiter=',')
    keyname = 'data\digitsY.dat'
    Y = loadtxt(keyname,delimiter=',')
    #Its a habit at this point
    print("Data loading time:", time()-startTime)
    scaler = MinMaxScaler() 
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2,shuffle=True, random_state=200282)
    #Pipeline for GrisSearch
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    
    print (np.max(X))
    print(np.min(X))
    '''clf = Pipeline([
        ('nn', MLPClassifier(learning_rate='adaptive',solver='sgd',batch_size='auto',max_iter=500,tol=1e-4,n_iter_no_change=10,validation_fraction=0.05,early_stopping=True,hidden_layer_sizes=[25])),
    ])
    
    learning_rate = array(range(25))/25+1/25
    print('Learning rate list:', learning_rate)
    nn_para = {
        'nn__learning_rate_init':learning_rate,
        'nn__activation':['logistic','identity','tanh'],
    }
    test = GridSearchCV(estimator=clf,param_grid=nn_para,cv=2,verbose=3,n_jobs=-1)'''
    test = MLPClassifier(learning_rate_init=0.68,activation= 'logistic',learning_rate='adaptive',solver='sgd',batch_size='auto',max_iter=500,tol=1e-4,n_iter_no_change=10,validation_fraction=0.05,early_stopping=True,hidden_layer_sizes=[25])
    start_train = time()
    test.fit(Xtrain,Ytrain)
    print('Training time: ', time()-start_train)
    #print('Best parameters:',test.best_params_)
    #print('Highest accuracy:', test.best_score_)
    y_pred=test.predict(Xtest)
    acc = accuracy_score(Ytest,y_pred)
    print("Accuracy on brand new data:",acc)