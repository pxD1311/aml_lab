from PIL import Image
import numpy as np
import mnist
import tensorflow as tf#tensor is vector generalized to higher dimension
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
def accuracy(cm):
    diagonal=cm.trace()
    elements=cm.sum()
    return diagonal/elements
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train=x_train.reshape((-1,28*28))
x_test=x_test.reshape((-1,28*28))
x_train=(x_train/256)
x_test=(x_test/256)
clf=MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(64,64))#64 neurons relu->learn non linear relation adam-> to train for neuralnetwork
clf.fit(x_train,y_train)
prediction=clf.predict(x_test)
acc=confusion_matrix(y_test,prediction)
print(accuracy(acc))
img=Image.open('/five.png')#enter the path of pic
data=list(img.getdata())
for i in range (len(data)):
    data[i]=255-data[i]
result=data
result=np.array(result)/256
p=clf.predict([result])

print(p)