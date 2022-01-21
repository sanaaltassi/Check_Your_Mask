from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# %%

#object detection (nesne algılama)
#mimarısını yükleme
prototxtPath = "data/deploy.prototxt.txt"
# sinir ağırlıkları yükledik
weightsPath = "data/res10_300x300_ssd_iter_140000.caffemodel"
# ağırlık ve mimarısını birleştirdik
net = cv2.dnn.readNet(prototxtPath, weightsPath)
# %%
#mask sınıflandırıcı yükledik
model = load_model("data/face_mask_detector.h5")
# %%
#foto yükleme
image = cv2.imread("data/shahd.jpeg")
(h, w) = image.shape[:2]
#shape input layer = shape image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
# threshold/confidence_base =90% (Hassas)
confidence_base = 0.90
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confidence_base:
        #yüz kesme
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        face = image[startY:endY, startX:endX]
        # göruntu işleme model eğitimiz gibi olsun
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        #dört shape olsun
        face = np.expand_dims(face, axis=0)
        # kesilen fotoğraf sınıflandırıcıya verildi
        (mask, withoutMask) = model.predict(face)[0]
        #büyük olan oranı ,maskeli= yeşil, maskesiz= kırmızı
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        #maskeli/maskesiz oranı yazdır
        label =label+str(max(mask, withoutMask) * 100)
        cv2.putText(image, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        # final fotoğraf boyutunu değiştir
image=cv2.resize(image,(int(image.shape[1])//3,int(image.shape[0])//3))
cv2.imshow("Output", image)
cv2.waitKey(0)

# %%
cap = cv2.VideoCapture(0)
while(True):

    _,image=cap.read()
    (h, w) = image.shape[:2]
    #shape input layer = shape image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    #threshold\confidence_base =90% aldakka
    confidence_base = 0.90
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_base:
            #yüz kesme
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = image[startY:endY, startX:endX]
            # göruntu işleme model eğitimiz gibi olsun
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            #dört shape olsun
            face = np.expand_dims(face, axis=0)
            # kesilen fotoğraf sınıflandırıcıya verildi
            (mask, withoutMask) = model.predict(face)[0]
            #büyük olan oranı ,maskeli= yeşil, maskesiz= kırmızı
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            #maskeli/maskesiz oranı yazdır
            label =label+str(max(mask, withoutMask) * 100)
            cv2.putText(image, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    # final fotoğraf boyutunu değiştir
    image=cv2.resize(image,(int(image.shape[1])//3,int(image.shape[0])//3))
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()