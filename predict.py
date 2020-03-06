from matplotlib import image
import matplotlib.pyplot as plt
import glob
import numpy as np
import sklearn.metrics as skm
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

imagenes = []
target = []
for i in range(1,101):
    image_ = np.float_(image.imread('train/{}.jpg'.format(i))[:,:,0].flatten())
    imagenes.append(image_[2000:3000])
    if i%2 == 0:
        target.append(1.)
    else:
        target.append(0.)
        
imagenes = np.array(imagenes)
target = np.array(target)
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(imagenes,target,train_size=0.8)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
vectores = vectores[:,ii]

x_train = x_train @ vectores
x_test = x_test @ vectores

def SVC(c,x_fit,x,y_fit,y):

    svm_ = svm.SVC(C = c)
    svm_.fit(x_fit[:,0:10],y_fit)

    f1 = skm.f1_score(y,svm_.predict(x[:,0:10]))

    return f1

c = np.logspace(-4,2,30)

f1_c = []

for element in c:
    f1_c.append(SVC(element,x_train,x_test,y_train,y_test))

max_ = np.argmax(f1_c)
best_c = c[max_]

best_svm = svm.SVC(C = best_c)
best_svm.fit(x_train[:,0:10],y_train)
best_f10 = skm.f1_score(y_test,best_svm.predict(x_test[:,0:10]), pos_label = 0)
best_f11 = skm.f1_score(y_test,best_svm.predict(x_test[:,0:10]), pos_label = 1)

imagenes_test = []
files_test = glob.glob("test/*.jpg")

for element in files_test:
    imagetest_ = np.float_(image.imread(element)[:,:,0].flatten())
    imagenes_test.append(imagetest_[2000:3000])

imagenes_test = scaler.transform(imagenes_test)
imagenes_test = imagenes_test @ vectores
predict_test = best_svm.predict(imagenes_test[:,0:10])


out = open("test/predict_test.csv", "w")
out.write("Name,Target\n")
for f, p in zip(files_test, predict_test):
    out.write("{},{}\n".format(f.split("/")[-1],p))

out.close()
