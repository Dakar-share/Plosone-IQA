# -*- coding: utf-8 -*-
import os
os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32"
import time
import cPickle
import numpy as np
import hickle as hkl    
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.utils.visualize_util import plot
from keras.layers.core import Layer
from keras.regularizers import l2
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from keras.callbacks import ModelCheckpoint

def load_data():
  """
  Load dataset and split data into training and test sets
  """
  #-------train part------
  X_train=np.load('data/train/train_image.npy')#(number，3,16,16)
  #-------test part-------
  X_test=np.load('data/test/test_image.npy')#(number，3,16,16)
  #---------train_labels-----
  Y_train=np.load('data/labels/train_label_patch.npy')
  #---------test_labels-----
  Y_test=np.load('data/labels/test_label_patch.npy')
  print 'X_train.shape: ',X_train.shape
  print 'X_test.shape: ',X_test.shape
  return (X_train,X_test,Y_train,Y_test)

def CNN_model(img_rows, img_cols, channel=3, num_class=None):

    input = Input(shape=(channel, img_rows, img_cols))
    conv1_7x7= Convolution2D(16,7,7,name='conv1/7x7',activation='relu')(input)#,W_regularizer=l2(0.0002)
    pool2_2x2= MaxPooling2D(pool_size=(2,2),strides=(1,1),border_mode='valid',name='pool2')(conv1_7x7)
    poll_flat = Flatten()(pool2_2x2)
    #MLP
    fc_1 = Dense(200,name='fc_1',activation='relu')(poll_flat)
    drop_fc = Dropout(0.5)(fc_1)
    out = Dense(1,name='fc_2',activation='sigmoid')(drop_fc)
    # Create model
    model = Model(input=input, output=out)
    # Load cnn pre-trained data 
    #model.load_weights('models/weights.h5')#NOTE 
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_absolute_error', metrics=['accuracy'])  
    return model 
if __name__ =='__main__':
  batch_size=100
  nb_epoch =200
  # TODO: Load training and test sets
  X_train, X_test, Y_train, Y_test = load_data()
  model = CNN_model(16, 16, 3)
  plot(model, to_file='model.png')
  checkpointer=ModelCheckpoint(filepath='models/weights.h5',monitor='val_loss',verbose=1,save_best_only=True)

  start_time = time.clock()
  model.fit(X_train,Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              callbacks=[checkpointer],
              validation_split=0.2
              )
  end_time = time.clock()
  
  print 'train time is: ',(end_time-start_time)
  print 'load best weight.......'
  model.load_weights('models/weights.h5')
  cPickle.dump(model,open("./models/model.pkl","wb"))#save best model+weight
  #if other code want use this final model run: model = cPickle.load(open("model.pkl","rb"))
  start_time = time.clock()
  print "make predictions......."
  predictions_test = model.predict(X_test, batch_size=batch_size, verbose=1)
  end_time = time.clock()
  print 'test time is: ',(end_time-start_time)
  print "make average....."
  predictions_test=np.asarray(predictions_test)
  result=[]
  b = np.load('./data/labels/test_label.npy')
  k = predictions_test.shape[0]/b.shape[0]
  for i in range(b.shape[0]):
    result.append(np.average(predictions_test[i*k:(i+1)*k,0]))
  np.save('./data/labels/test_result.npy',result)#predict result
  print "Done...."
