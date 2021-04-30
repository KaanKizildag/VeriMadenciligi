from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
import VeriYukle
import numpy as np
print(' Yapay sinir ağları '.upper().center(50, '-'))
x_train, x_test, y_train, y_test = VeriYukle.egitimTestVeriSeti()

ysa = MLPClassifier(random_state=1, max_iter=1000)
ysa.fit(x_train, y_train)
y_pred = ysa.predict(x_test)
cm = confusion_matrix(y_true = y_test, y_pred = y_pred)

print('Doğruluk matrisi'.capitalize().title())
print(cm)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

print('Doğruluk değerleri'.upper().center(50, '-'))
print(f"FP:{FP.sum()} FN:{FN.sum()} TP:{TP.sum()} TN:{TN.sum()}")
dogruluk = (TP.sum() + TN.sum()) /  (TP.sum() + TN.sum() + FN.sum() + FP.sum()) * 100
print(f'Doğruluk: % {dogruluk}')

print(classification_report(y_test, y_pred))
