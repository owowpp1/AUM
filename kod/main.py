import pprint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import seaborn as sn

import time


wartosci={
	'AB':{
		0:'Aa',
		1:'Bb'
	},
	'AaBb':{
		0:'A',
		1:'a',
		2:'B',
		3:'b'
	}
}

def wczytaj_dane(wariant='AaBb'):
	df = pd.read_csv (f'../data/cechy_{wariant}.csv', delimiter=';')
	X=df.iloc[:,1:-1]
	Y=df.iloc[:,-1]
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	
	print("Rozmiar danych uczonych:",len(X_train))
	print("Rozmiar danych testowych:",len(X_test))

	return {
		'df':df,
		'X':X,
		'Y':Y,
		'X_train':X_train,
		'X_test':X_test,
		'y_train':y_train,
		'y_test':y_test
	}

def plotuj_cechy(df, wariant='AaBb'):
	plt.clf()
	if wariant=='AaBb':
		A=df.loc[df['wartość']==0]
		a=df.loc[df['wartość']==1]
		B=df.loc[df['wartość']==2]
		b=df.loc[df['wartość']==3]
		A=A.iloc[:,1:-1]
		a=a.iloc[:,1:-1]
		B=B.iloc[:,1:-1]
		b=b.iloc[:,1:-1]
		literki=[A,a,B,b]
		nazwy=["A", 'a', "B", 'b']
		width=0.2
		X=A.columns
	else:
		Aa=df.loc[df['wartość']==0]
		Bb=df.loc[df['wartość']==1]
		Aa=Aa.iloc[:,1:-1]
		Bb=Bb.iloc[:,1:-1]
		literki=[Aa,Bb]
		nazwy=["Aa", 'Bb']
		width=0.3
		X=Aa.columns
	ind=np.arange(len(X))
	for nr,dane in enumerate(literki):
		Y=[]
		for i,a in enumerate(X):
			Y.append(0)
		for x,row in dane.iterrows():
			for i,kolumna in enumerate(X):
				Y[i]+=row[kolumna]

		plt.bar(ind+nr*width,Y,width, label=f'Literka: {nazwy[nr]}')
		plt.xlabel('cecha')
		plt.ylabel('liczba wystąpień w sumie')
		plt.xticks(ind+width/2, X, rotation=45, ha="right")
	plt.legend()
	if wariant=='AaBb':
		plt.title('Sumaryczna liczba wystąpień cech dla liter A, a, B oraz b')
	else:
		plt.title('Sumaryczna liczba wystąpień cech dla liter Aa oraz Bb')
	plt.savefig(f'wystapienia_cech_sum_{wariant}.jpg', bbox_inches = "tight")

def plotuj_wyniki(wyniki, title='Dokładność algorytmów', ylabel='dokładność', leg_loc='lower left'):
	width=0.3
	plt.clf()
	for nr,wariant in enumerate(wyniki):
		X=list(wyniki[wariant].keys())
		Y=list(wyniki[wariant].values())
		ind=np.arange(len(X))
		plt.bar(ind+nr*width,Y,width, label=f'Wariant: {wariant}')
		plt.xlabel('algorytm')
		plt.ylabel(ylabel)
		plt.xticks(ind+width/2, X, rotation=45, ha="right")
	plt.legend(loc=leg_loc)
	plt.title(title)
	plt.savefig(f'{ylabel}.jpg', bbox_inches = "tight")

def plotuj_CM(y_test, preds, wariant, nazwa):
	cm = confusion_matrix(y_test, preds)
	plt.clf()
	ax=plt.subplot()
	sn.heatmap(cm, annot=True, fmt='g', ax=ax)
	ax.set_xlabel('przewidziane wartości')
	ax.set_ylabel('prawdziwe wartości') 
	ax.set_title(f'Tablica Pomyłek {nazwa} wariant {wariant}')
	if wariant=='AaBb':
		ax.xaxis.set_ticklabels(['A', 'a', 'B','b'])
		ax.yaxis.set_ticklabels(['A', 'a', 'B','b'])
	else:
		ax.xaxis.set_ticklabels(['Aa', 'Bb'])
		ax.yaxis.set_ticklabels(['Aa', 'Bb'])
	plt.savefig(f'CM/{nazwa}_{wariant}.jpg')
	return cm

def plotuj_f1(f1s):
	# width=0.6
	for wariant in f1s:
		X=list(f1s[wariant].keys())
		plt.clf()
		for nr in range(len(list(f1s[wariant].values())[0])):
			width=0.6/len(list(f1s[wariant].values())[0])
			Y=list(f1s[wariant].values())
			Y=[a[nr] for a in Y]
			
			ind=np.arange(len(X))
			plt.bar(ind+nr*width,Y,width, label=f'Literka: {wartosci[wariant][nr]}')
			plt.xlabel('algorytm')
			plt.ylabel('f1')
			plt.xticks(ind+width/2, X, rotation=45, ha="right")
		plt.legend(loc='lower left')
		plt.title(f'Współczynniki f1 wariant {wariant}')
		plt.savefig(f'f1/{wariant}.jpg', bbox_inches = "tight")

def przelicz_CM(cm):
	TPs=[cm[i][i] for i in range(len(cm))]		#True Positive
	TNs=[0 for i in range(len(cm))]				#True Negative
	for i,row in enumerate(cm):
		cm2=cm.tolist()
		cm2.pop(i)
		for row2 in cm2:
			row2.pop(i)
			TNs[i]+=sum(row2)
			
	FPs=[sum(row)-row[i] for i,row in enumerate(cm)]						#False Positive

	FNs=[sum([row2[i] for row2 in cm])-row[i] for i,row in enumerate(cm)]	#False Negative
	
	precision=[]
	recall=[]
	f1=[]
	for i in range(len(cm)):
		precision.append(TPs[i]/(TPs[i]+FPs[i]))
		recall.append(TPs[i]/(TPs[i]+FNs[i]))
		f1.append(2*(1/(1/precision[i]+1/recall[i])))
	
	return {
		'TPs':TPs,
		'TNs':TNs,
		'FPs':FPs,
		'FNs':FNs,
		'prec':precision,
		'rec':recall,
		'f1':f1
	}

def przelicz_MAN(y_pred, y_test):
	suma=0
	for i,val in enumerate(y_pred):
		suma+=abs(val-y_test[i])
	return suma/len(y_pred)


def trenuj_xgboost(dane, wariant='AaBb', num_round=20):

	dtrain = xgb.DMatrix(dane['X_train'], label=dane['y_train'])
	dtest = xgb.DMatrix(dane['X_test'], label=dane['y_test'])

	parameters = {
		'eta': 0.3,  
		'objective': 'multi:softprob',  # error evaluation for multiclass tasks
		'num_class': 2,  # number of classes to predic
		'max_depth': 3  # depth of the trees in the boosting process
		} 
	if wariant=='AaBb':
		parameters['num_class']=4

	bst = xgb.train(parameters, dtrain, num_round)
	preds = bst.predict(dtest)

	best_preds = np.asarray([np.argmax(line) for line in preds])
	dobrosc=sum(list(best_preds)[i]==list(dane['y_test'])[i] for i in range(len(list(best_preds))))/len(list(best_preds))
	print(f"dobrość: {dobrosc}")
	return {
		'preds':list(best_preds),
		'dobrosc':dobrosc
	}

def trenuj_KNeighbors(dane):
	knn = KNeighborsClassifier()
	knn.fit(dane['X_train'], dane['y_train'])
	y_pred = knn.predict(dane['X_test'])

	print(list(y_pred))
	print(list(dane['y_test']))
	dobrosc=sum(list(y_pred)[i]==list(dane['y_test'])[i] for i in range(len(list(y_pred))))/len(list(y_pred))
	print(f"dobrość: {dobrosc}")
	return {
		'preds':list(y_pred),
		'dobrosc':dobrosc
	}

def trenuj_RandomForest(dane):
	rfc = RandomForestClassifier()
	rfc.fit(dane['X_train'], dane['y_train'])
	y_pred = rfc.predict(dane['X_test'])

	print(list(y_pred))
	print(list(dane['y_test']))
	dobrosc=sum(list(y_pred)[i]==list(dane['y_test'])[i] for i in range(len(list(y_pred))))/len(list(y_pred))
	print(f"dobrość: {dobrosc}")
	return {
		'preds':list(y_pred),
		'dobrosc':dobrosc
	}

warianty=['AB', 'AaBb']
wyniki={}
predykcje={}
czasy={}
f1s={}
MANs={}
for wariant in warianty:
	print(f'\n=====WARIANT: {wariant}=====\n')
	wyniki[wariant]={}
	czasy[wariant]={}
	f1s[wariant]={}
	MANs[wariant]={}

	dane=wczytaj_dane(wariant)

	predykcje[wariant]=[0 for i in range(len(dane['X_test']))]

	plotuj_cechy(dane['df'], wariant)

	print('=====XGBOOST=====')
	xgpreds=[0 for i in range(len(dane['X_test']))]
	planowane_powtorzenia=[20, 100, 1000]
	for powtorzenia in planowane_powtorzenia:
		czas=time.time()
		wynik=trenuj_xgboost(dane, wariant, powtorzenia)
		czasy[wariant][f'xgboost{powtorzenia}']=time.time()-czas
		wyniki[wariant][f'xgboost{powtorzenia}']=wynik['dobrosc']
		xgpreds=[xgpreds[i]+wynik['preds'][i] for i in range(len(wynik['preds']))]
		cm=plotuj_CM(dane['y_test'], wynik['preds'], wariant, f'xgboost{powtorzenia}')
		przeliczoneCM=przelicz_CM(cm)
		f1s[wariant][f'xgboost{powtorzenia}']=przeliczoneCM['f1']
		MANs[wariant][f'xgboost{powtorzenia}']=przelicz_MAN(wynik['preds'], list(dane['y_test']))
	
	xgpreds=[int(val/len(planowane_powtorzenia)+0.5) for val in xgpreds]
	print(xgpreds)
	dobrosc=sum(list(xgpreds)[i]==list(dane['y_test'])[i] for i in range(len(list(xgpreds))))/len(list(xgpreds))
	print(f"dobrość (średnia): {dobrosc}")
	wyniki[wariant]['xgboostAverage']=dobrosc
	cm=plotuj_CM(dane['y_test'], xgpreds, wariant, 'xgboostave')
	przeliczoneCM=przelicz_CM(cm)
	f1s[wariant][f'xgboostAverage']=przeliczoneCM['f1']
	MANs[wariant]['xgboostAverage']=przelicz_MAN(xgpreds, list(dane['y_test']))
	

	predykcje[wariant]=[predykcje[wariant][i]+xgpreds[i] for i in range(len(dane['X_test']))]

	print('=====KNeighbors=====')
	czas=time.time()
	wynik=trenuj_KNeighbors(dane)
	czasy[wariant][f'KNeighbors']=time.time()-czas
	wyniki[wariant]['KNeighbors']=wynik['dobrosc']

	predykcje[wariant]=[predykcje[wariant][i]+wynik['preds'][i] for i in range(len(dane['X_test']))]

	cm=plotuj_CM(dane['y_test'], wynik['preds'], wariant, 'KNeighbors')
	przeliczoneCM=przelicz_CM(cm)
	f1s[wariant]['KNeighbors']=przeliczoneCM['f1']
	MANs[wariant]['KNeighbors']=przelicz_MAN(wynik['preds'], list(dane['y_test']))

	print('=====RANDOM FOREST=====')
	czas=time.time()
	wynik=trenuj_RandomForest(dane)
	czasy[wariant][f'RandomForest']=time.time()-czas
	wyniki[wariant]['RandomForest']=wynik['dobrosc']
	
	predykcje[wariant]=[predykcje[wariant][i]+wynik['preds'][i] for i in range(len(dane['X_test']))]

	cm=plotuj_CM(dane['y_test'], wynik['preds'], wariant, 'RandomForest')
	przeliczoneCM=przelicz_CM(cm)
	f1s[wariant][f'RandomForest']=przeliczoneCM['f1']
	MANs[wariant]['RandomForest']=przelicz_MAN(wynik['preds'], list(dane['y_test']))

	predykcje[wariant]=[int(val/3+0.5) for val in predykcje[wariant]]
	dobrosc=sum(list(predykcje[wariant])[i]==list(dane['y_test'])[i] for i in range(len(list(predykcje[wariant]))))/len(list(predykcje[wariant]))
	print(f"dobrość (średnia ogólna): {dobrosc}")
	wyniki[wariant]['Średnia']=dobrosc

	cm=plotuj_CM(dane['y_test'], predykcje[wariant], wariant, 'Average')
	przeliczoneCM=przelicz_CM(cm)
	f1s[wariant][f'Średnia']=przeliczoneCM['f1']
	MANs[wariant]['Średnia']=przelicz_MAN(predykcje[wariant], list(dane['y_test']))

print('\n\n=====WYNIKI=====')
pprint.pprint(wyniki, sort_dicts=False)
print('=====CZASY=====')
pprint.pprint(czasy, sort_dicts=False)
print('=====F1S=====')
pprint.pprint(f1s, sort_dicts=False)
print('=====MANs=====')
pprint.pprint(MANs, sort_dicts=False)

plotuj_wyniki(wyniki)

plotuj_wyniki(czasy, 'Czasy działania algorytmów', 'czas(ms)', 'upper left')

plotuj_f1(f1s)

plotuj_wyniki(MANs, 'Mean Absolute Error', 'współczynnik MAN', 'upper left')
