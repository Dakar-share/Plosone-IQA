import numpy as np
a=np.load('test_label.npy')
with open('test_label.txt', 'w') as f:
    for ind in range(a.shape[0]):
        str_write =str('%03f'%(float(a[ind]))) +'\n'  #note: int or float 
	f.write(str_write)
a=np.load('test_result.npy')
with open('test_result.txt', 'w') as f:
    for ind in range(a.shape[0]):
        str_write =str('%03f'%(float(a[ind]))) +'\n'  #note: int or float 
	f.write(str_write)
