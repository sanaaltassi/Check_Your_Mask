#%%
from sklearn.model_selection import train_test_split
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.models import Model,Sequential
from sklearn.metrics import classification_report
#%%
path="data/with_mask/*.jpg"
# fotoğraf listesini başlatmak
list_images1 = []
for i in glob(path):
  image1 = cv2.imread(i)

  image1=cv2.resize(image1,(224,224))
  image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)

  list_images1.append(image1)

images1=np.array(list_images1)
images1 = images1.astype('float32') / 255.0  #  normalize [0,1]
#%%
plt.imshow(images1[0])
plt.show()
#%%
path="data/without_mask/*.jpg"
# initialize the list of images
list_images2 = []
for i in glob(path):
  # image2 = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
  image2 = cv2.imread(i)
  image2=cv2.resize(image2,(224,224))
  image2=cv2.cvtColor(image2,cv2.COLO)

  list_images2.append(image2)
images2=np.array(list_images2)
images2 = images2.astype('float32') / 255.0  #  resize to[0,1]
#%%
plt.imshow(images2[0])
plt.show()
#%%
data = np.concatenate((images1, images2), axis=0)
labels =  np.concatenate((np.ones(images1.shape[0],dtype=int),np.zeros((images2.shape[0]), dtype=int)), axis=0)
#%%

x_train, x_val, y_train, y_val = train_test_split( data, labels, test_size=0.33, random_state=42)
#%%
# bu mimarısı cnn ağı için kullandık , hyper parametrs değiştirdikten , bu parametreleri en iyi sonuç göstermiş
model = Sequential()

model.add(Conv2D(input_shape=(224,224,3),filters=16,kernel_size=(2,2),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
# conv , maxpool, bath normalization şunlar ağını mimarısı ve 3 sefer ekledik
# drop out layeri overfiting problemi çözmek için
model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# standart layerler(katmanlar)
model.add(Flatten())
model.add(Dense(1024,activation="relu"))
#sadce iki classımız var buyüzden  en son layerde sadece bir node kullandık
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

#%%
# eğitim yap
model.fit(x_train , y_train , batch_size=30, epochs=16, verbose=1 )
#%%
x2=model.evaluate(x_train,y_train)
print("training data = ",x2)
x3=model.evaluate(x_val,y_val)
print("val data = ",x3)
#%%

# model.save("face_mask_detector.h5")



