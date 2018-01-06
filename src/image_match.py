from keras.applications                import VGG16
from keras.models                      import Model
from keras.applications.imagenet_utils import preprocess_input
from scipy.misc                        import imread, imresize
from sklearn.neighbors                 import KNeighborsClassifier
import numpy as np
import pickle
import cv2
import os


CLASSIFIER_MODEL = 'models/model.p'
CASCADE_MODEL    = 'models/haarcascade_frontalface_alt.xml'


class ImageMatch(object):

    def __init__(self):
        self.__make_vgg16_model()
        self.__cascade = cv2.CascadeClassifier(CASCADE_MODEL)
        if os.path.isfile(CLASSIFIER_MODEL):
            self.__load_classifier()
        else:
            self.__train_classifier()

    def __make_vgg16_model(self):
        vgg16 = VGG16(weights='imagenet', include_top=False)
        self.__model = Model(
            inputs  = [vgg16.input], 
            outputs = [vgg16.get_layer('block5_pool').output]
        )

    def __vgg16_predict(self, image):
        image = imresize(image, (224,224)).astype(np.float32)
        image = preprocess_input(image)
        image = np.reshape(image, (1,224,224,3))
        label = self.__model.predict(image)
        label = label.ravel()
        return label

    def __generate_training_data(self):
        print('Training...')
        features, labels = [], []
        dirlist = os.listdir('data/train')
        images  = [f for f in dirlist if 'jpg' in f] 
        for image in images:
            filename = os.path.splitext(image)[0]
            label    = ''.join(i for i in filename if not i.isdigit())
            labels.append(label)
            feature = imread('data/train/{}.jpg'.format(filename))
            feature = self.__vgg16_predict(feature)
            features.append(feature)
        return features, labels

    def __save_classifier(self):
        fp = open(CLASSIFIER_MODEL, 'wb')
        pickle.dump(self.__classifier, fp)

    def __load_classifier(self):
        fp = open(CLASSIFIER_MODEL, 'rb')
        self.__classifier = pickle.load(fp)

    def __train_classifier(self):
        features, labels = self.__generate_training_data()
        count = len(np.unique(labels))
        self.__classifier = KNeighborsClassifier(n_neighbors=count)
        self.__classifier.fit(features, labels)
        self.__save_classifier()
   
    def __detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.__cascade.detectMultiScale(
            gray, 
            scaleFactor  = 1.1, 
            minNeighbors = 5
        )
        x,y,w,h = faces[0]
        image = image[y:y+h,x:x+w]
        return image

    def match(self, image):
        image   = self.__detect_face(image) 
        feature = self.__vgg16_predict(image)
        label   = self.__classifier.predict([feature])
        return label[0]

