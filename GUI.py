from tkinter import *
from tkinter import filedialog
from tensorflow.keras.models import load_model
from PIL import ImageTk,Image
import numpy as np
import tkinter.font as tkFont
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

load=Image
root=Tk()

print("[loading face detector model")
prototxtPath = "data/deploy.prototxt.txt"
weightsPath = "data/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)
#%%
confidence_base=0.90

# load the face mask detector model from disk
print("loading face mask detector model")
model = load_model("data/face_mask_detector.h5")#face_mask_detector.h5
def fileDialog():
    filename = filedialog.askopenfilename(initialdir="\Desktop", title="Select A File", filetype=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    global im
    im=cv2.imread(filename)
    load = Image.open(filename)

    load2=load.resize((300,300))
    render = ImageTk.PhotoImage(load2)
    img = Label(root, image=render)
    img.image = render
    img.place(relx=0.5,
                       rely=0.5,
                       anchor='center')
    # img.place(x=0, y=0)
def predc():
    image = im.copy()

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("computing face detections")
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > confidence_base:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            face = np.expand_dims(face, axis=0)
            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(300, 300))
    image = Image.fromarray(image)
    render = ImageTk.PhotoImage(image)
    img = Label(root, image=render)
    img.image = render
    img.place(relx=0.5,
                       rely=0.5,
                       anchor='center')

root.title("Python Tkinter Dialog Widget")
root.minsize(640, 400)

Button1 = Button(root,text="upload image", command=fileDialog)
Button1.place(relx=0.20,rely=0.93)

Button2 = Button(root,text="predict result", command=predc)
Button2.place(relx=0.70,rely=0.93)

fontStyle = tkFont.Font(family="Lucida Grande", size=20)

root.mainloop()

