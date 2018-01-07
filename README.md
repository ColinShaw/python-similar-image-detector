# Face Matching Service

This is a fun little matching service built using Keras, 
TensorFlow, scikit-learn and OpenCV.  OpenCV is used for
Haar cascade face region identification for matching, 
the training data is pre-cropped faces. The cropped faces 
are fed into VGG16 with the dense classification layers 
removed.  The resulting flattened convolutional vector 
has principal component analysis applied to reduce the 
space, the result of which is fed to a linear support
vector machine classifier to obtain class identification.

What you do is drop some training data in `/data/train` 
using a common name with numeric suffix for the repeats 
of a given class.  These will all be mapped to the 
non-numeric part of the file name as a label.  You can feed
a generic catch-all class like `none` for a wide range of
images that do not match a particular class.  The reason for
doing the PCA before the classifier is to reduce the
dimensionality of the VGG16 vector to something more meaningful
to the classifier.  This makes a huge difference when there 
is a catch-all class (as opposed to forcing only defined
classes).  

When the app first runs, it will notice a lack of the 
pre-computed model and build one from the files present.  Next 
time it runs it will simply use the existing classifier.  Both 
the classifier and decomposition model are stored in the 
`/models` directory.  Just delete it if you want to refresh 
the classifier model with new data.

To perform a match, you just send an image by `POST` to
the service.  Here are some examples based on the data that
this comes with:

```
curl -X POST -F 'image=@data/test/norton.jpg' http://localhost:5000/match
curl -X POST -F 'image=@data/test/hopkins.jpg' http://localhost:5000/match
curl -X POST -F 'image=@data/test/cruise.jpg' http://localhost:5000/match
curl -X POST -F 'image=@data/test/roberts.jpg' http://localhost:5000/match
```

Given the training data, the Edward Norton and Anthony Hopkins matches
should return the proper class, whereas Tom Cruise and Julia Roberts
should return the `none` class.


