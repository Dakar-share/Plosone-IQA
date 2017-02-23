import numpy as np
import hickle as hkl
import scipy.misc
from sklearn.cross_validation import StratifiedShuffleSplit,ShuffleSplit,train_test_split
from pylearn2.utils.rng import make_np_rng
import os
import shutil
def split_train_test(rng,label):
   train_label=[]
   train_image=[]
   test_label=[]
   test_image=[]
   image_all_lcn=np.load('./data/image_all/image_all_lcn.npy')#note lcn is float
   #image_all_lcn = hkl.load('./data/image_all/image_all.hkl')#this is no normalization
   print 'image_all.shape',image_all_lcn.shape
   sss = ShuffleSplit(image_all_lcn.shape[3]/5, n_iter=1, test_size=0.2, random_state=rng)#note:this dataset has 25 diffent pic,each pic has 5 same blur pic
   print 'Note: it only use TID2013_blur dataset!!!,diffent pic is: ',image_all_lcn.shape[3]/5
   for train_index,test_index in sss:
      print 'train_index',train_index
      print 'test_index' ,test_index
      train_index = train_index[np.argsort(train_index)]
      test_index = test_index[np.argsort(test_index)]
      print 'train_index_sort',train_index
      print 'test_index_sort' ,test_index
      for i in train_index:
          for j in range(5):#note eatch diffent picture have 5 picture
              train_image.append(image_all_lcn[:,:,:,5*i+j])
              train_label.append(label[5*i+j])
      train_image = np.asarray(train_image)
      train_image = np.rollaxis(train_image,0,4)#(number,3,h,w)->(3,h,w,number)
      print 'train_image.shape: ',train_image.shape
      np.save('./data/labels/train_label.npy',train_label)
      np.save('./data/train/train_image.npy',train_image)
      for i in test_index:
          for j in range(5):#note eatch diffent picture have 5 picture
              test_image.append(image_all_lcn[:,:,:,5*i+j])
              test_label.append(label[5*i+j])
      test_image = np.asarray(test_image)
      test_image = np.rollaxis(test_image,0,4)#(number,3,h,w)->(3,h,w,number)
      print 'test_image.shape: ',test_image.shape
      np.save("./data/labels/test_label.npy",test_label)
      np.save('./data/test/test_image.npy',test_image)
      print 'Split Done.....'

if __name__ == '__main__':
    if os.path.isdir('./data/train')==False:
       os.makedirs('./data/train')
    if os.path.isdir('./data/test') ==False:
       os.makedirs('./data/test')
    label=np.load('./data/labels/label_all.npy')
    print 'label_all.shape',label.shape
    rng = [2016,12,17]#2016 12 17
    rng = make_np_rng(None, rng, which_method='uniform')
    split_train_test(rng,label)


