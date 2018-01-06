# Face Matching Service

This is a fun little matching service built using Keras, 
TensorFlow, scikit-learn and OpenCV.  OpenCV is used for 
actual image matching using a Haar cascade to detect 
face regions.  These are fed into VGG16 with the 
classification layers removed.  The resulting flattened 
convolutional vector is fed into a K Nearest Neighbor
classifier to obtain the identification.  

What you do is drop some training data in `/data/train` 
using a common name with numeric suffix for the repeats 
of a given class.  These will all be mapped to the 
non-numeric part of the file name as a label.  The number of
nearest neighbors is based on the number of unique labels.  When 
the app first runs, it will notice a lack of the model and 
build one from the files present.  Next time it runs it
will simply use the existing classifier.  The classifier
is stored in the `/models` directory.  Just delete it if
you want to refresh the classifier model.

To perform a match, you just send an image by `POST` to
the service.  Here are some examples based on the data that
this comes with:

```
curl -X POST -F 'image=@data/test/norton.jpg' http://localhost:5000/match
curl -X POST -F 'image=@data/test/hopkins.jpg' http://localhost:5000/match
```

Have fun!

