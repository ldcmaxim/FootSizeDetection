import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import skimage as sk
from skimage import img_as_ubyte
from model import *
from utils import *
from flask import Flask
from flask_restful import Resource, Api, reqparse,request
import json
import tensorflow as tf

app = Flask(__name__)

TARGET_SIZE = (256,256)

model = unet(pretrained_weights="./weights.hdf5")
graph = tf.get_default_graph()


def preprocesss(img):

    img = trans.resize(img, TARGET_SIZE)
    img = np.reshape(img, img.shape + (1,)) if (not False) else img
    img = np.reshape(img, (1,) + img.shape)
    return img


def get_mask(width,height,npyfile):

    for i,item in enumerate(npyfile):

        img = item[:,:,0]
        img = trans.resize(img, (width, height))

        return img


def predict_unet(img):


    width, height = img.shape

    img = preprocesss(img)


    with graph.as_default():
        mask = model.predict(img)


    mask = get_mask(width, height, mask)

    mask = img_as_ubyte(mask)


    return mask


def pipeline(image):


    mask = predict_unet(image)


    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)


    if not image is None and not mask is None:

        paper_width_distance, unsheared = unshear(mask, image)


        pixel_per_metric = paper_width_distance / 8.3


        auto = auto_white(unsheared)

        mask_final = generate_mask(auto)

        gray_mask_final = cv2.cvtColor(mask_final, cv2.COLOR_BGR2GRAY)


        contours, hierarchy = cv2.findContours(gray_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key=cv2.contourArea)
        x, y, w, foot_height = cv2.boundingRect(c)

        cropped = crop_half(mask_final)

        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key=cv2.contourArea)
        x, y, foot_width, h = cv2.boundingRect(c)


        Data = {"dimensions" : {"width": str(foot_width / pixel_per_metric) , "height": str(foot_height / pixel_per_metric)}}

        return json.dumps(Data)




@app.route('/getfile', methods=['POST'])
def getfile():

    json = ""

    try:

        file = request.files['image']

        image = io.imread(file,as_gray=True)

        image = img_as_ubyte(image)

        json = pipeline(image)



    except:

        return  "Some error occured. Please try again !"


    return json


if __name__ == '__main__':


    app.run(host = '127.0.0.1',debug=True)






























