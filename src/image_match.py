from keras.applications    import VGG16, imagenet_utils
from keras.models          import Model
from scipy.misc            import imread, imresize
from sklearn.svm           import LinearSVC
from sklearn.decomposition import PCA
from os.path               import isfile
import numpy as np
import pickle
import cv2
import os


CLASSIFIER_MODEL    = 'models/classifier.p'
DECOMPOSITION_MODEL = 'models/decomposition.p'
CASCADE_MODEL       = 'models/haarcascade_frontalface_alt.xml'


class ImageMatch(object):

    def __init__(self):
        self.__make_vgg16_model()
        self.__cascade = cv2.CascadeClassifier(CASCADE_MODEL)
        if isfile(CLASSIFIER_MODEL) and isfile(DECOMPOSITION_MODEL):
            self.__load_classifier()
            self.__load_decomposition()
        else:
            self.__train_classifier()
            self.__save_classifier()
            self.__save_decomposition()

    def __make_vgg16_model(self):
        vgg16 = VGG16(weights='imagenet', include_top=False)
        self.__model = Model(
            inputs  = [vgg16.input], 
            outputs = [vgg16.get_layer('block5_pool').output]
        )

    def __save_classifier(self):
        fp = open(CLASSIFIER_MODEL, 'wb')
        pickle.dump(self.__classifier, fp)

    def __load_classifier(self):
        fp = open(CLASSIFIER_MODEL, 'rb')
        self.__classifier = pickle.load(fp)

    def __save_decomposition(self):
        fp = open(DECOMPOSITION_MODEL, 'wb')
        pickle.dump(self.__decomposition, fp)

    def __load_decomposition(self):
        fp = open(DECOMPOSITION_MODEL, 'rb')
        self.__decomposition = pickle.load(fp)

    def __vgg16_predict(self, image):
        image = imresize(image, (224,224)).astype(np.float32)
        image = imagenet_utils.preprocess_input(image)
        image = np.reshape(image, (1,224,224,3))
        label = self.__model.predict(image)
        return label.ravel()

    def __generate_training_data(self):
        features, labels = [], []
        dirlist = os.listdir('data/train')
        images  = [f for f in dirlist if 'jpg' in f] 
        for image in images:
            filename = os.path.splitext(image)[0]
            label = ''.join(i for i in filename if not i.isdigit())
            labels.append(label)
            feature = imread('data/train/{}.jpg'.format(filename))
            feature = self.__vgg16_predict(feature)
            features.append(feature)
        return features, labels

    def __train_classifier(self):
        print('Training...')
        features, labels = self.__generate_training_data()
        self.__decomposition = PCA()
        self.__classifier    = LinearSVC()
        self.__decomposition.fit(features)
        features = self.__decomposition.transform(features)
        self.__classifier.fit(features, labels)
   
    def __detect_face(self, image):
        gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.__cascade.detectMultiScale(gray)
        x,y,w,h = faces[0]
        return image[y:y+h,x:x+w]

    def match(self, image):
        image   = self.__detect_face(image) 
        feature = self.__vgg16_predict(image)
        feature = self.__decomposition.transform([feature])
        label   = self.__classifier.predict(feature)
        return label[0]

