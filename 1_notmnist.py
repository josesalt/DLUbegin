
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 1
# ------------
# 
# The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
# 
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.

# In[2]:

url = 'http://yaroslavvb.com/upload/notMNIST/'

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labelled A through J.

# In[ ]:

num_classes = 10

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


# ---
# Problem 1
# ---------
# 
# Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
# 
# ---

# In[ ]:

def showImage(data_folder, numImag):
  """Load the data for a single letter label."""  
  for folder in data_folder:
    contador = 1
    image_files = os.listdir(folder)
    for image in os.listdir(folder):	
      image_file = os.path.join(folder, image)
      display(Image(image_file))
      contador = contador + 1
      if contador > numImag:
        break
  
showImage(train_folders,1)
showImage(test_folders,1)


# Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.
# 
# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. 
# 
# A few images might not be readable, we'll just skip them.

# In[ ]:

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  image_index = 0
  print(folder)
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


# ---
# Problem 2
# ---------
# 
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
# 
# ---
from random import randint 

def showImagePlot(filename):
  """Load the data for a single letter label."""
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  
  for pickleFile in os.listdir(root):
      if (pickleFile.endswith('.pickle')):        
        pickle_file = os.path.join(root, pickleFile)    
        ax = pickle.load(open(pickle_file,  'rb') )
        imgRand = randint(0,len(ax))
        plt.imshow(ax[imgRand])
        plt.show()

def showImagePlot2(source):
  """Load the data for a single letter label."""  
  for i in range(0,10):
    ax = pickle.load(open(source[i],  'rb') )
    imgRand = randint(0,len(ax))
    plt.imshow(ax[imgRand])
    plt.show()    

#showImagePlot(train_filename)
##showImagePlot(test_filename)

#showImagePlot2(train_datasets)
#showImagePlot2(test_datasets)


# ---
# Problem 3
# ---------
# Another check: we expect the data to be balanced across classes. Verify that.
# 
# ---

# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.
# 
# Also create a validation dataset for hyperparameter tuning.



# In[ ]:

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def calcResumen(source, dato):
  print('Resumen del carácter: ',dato)
  print('Full dataset tensor:', source.shape)
  print('Mean:', np.mean(source))
  print('Standard deviation:', np.std(source))
  

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                              
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
        
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  calcResumen(train_dataset, train_labels)    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def showSetlenght(source, tipo):
  print('Cantidad de instancias en: ', tipo, ' = ', len(source))
  
  y = np.bincount(source)
  ii = np.nonzero(y)[0]
  x = np.vstack((ii,y[ii])).T
  print(x)

showSetlenght(train_labels, 'train set')
showSetlenght(valid_labels, 'valid set')
showSetlenght(test_labels, 'test set')

# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

# In[ ]:

np.random.seed(133)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  calcResumen(shuffled_dataset, shuffled_labels)
  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)

showSetlenght(train_labels, 'train set after shuffle')
showSetlenght(valid_labels, 'valid set after shuffle')
showSetlenght(test_labels, 'test set after shuffle')

# ---
# Problem 4
# ---------
# Convince yourself that the data is still good after shuffling!
# 
# ---

# Finally, let's save the data for later reuse:

# In[ ]:

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


# In[ ]:

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

## 0 trainSet, 1 testSet, 2 validSet
def typeCompare(typeValue):
  if (typeValue ==0): 
    return 'validSet', 'trainSet'
  if (typeValue ==1):
    return 'validSet', 'testSet'
  if (typeValue ==2):
    return 'testSet', 'trainSet'

def findInstInSet(sourceSet, compSet):
  import collections
##  for instance in souceSet:
##    for instanceCompset in compSet:

##  d = collections.OrderedDict()
##  for a in sourceSet:
##    t = tuple(a)
##    if t in compSet:
##      d[t] += 1
##    else:
##        d[t] = 1
##
##  result = []
##  for (key, value) in d.items():
##    result.append(list(key) + [value])
##
##  B = numpy.asarray(result)
  sourceSet.flags.writeable=False
  compSet.flags.writeable=False
  dup_table={}
  for idx,img in enumerate(sourceSet):
    h = hash(img.data)
    if h in dup_table and (compSet[dup_table[h]].data == img.data):
       print ('Duplicate image: %d matches %d' % (idx, dup_table[h]))
    dup_table[h] = idx

findInstInSet(valid_dataset, train_dataset)
findInstInSet(valid_dataset, test_dataset)

findInstInSet(test_dataset, train_dataset)

def findInstInSet2(sourceSet, compSet, typeComp):
  contador = 0
  sourceSet.flags.writeable = False
  compSet.flags.writeable = False


  import timeit
  start_time = timeit.default_timer()
  
  
  for instance in sourceSet:
    first = hash(instance.data)
    for instanceCompset in compSet:
      second = hash(instanceCompset.data)
      if first == second:
        ++contador

  elapsed = timeit.default_timer() - start_time
          
  print ("Duplicados en %d :%d", typeCompare(typeComp),contador )
  print ("Tiempo de procesamiento: ", elapsed)


def imprimirInstancias():
  contador = 0
  train_dataset.flags.writeable = False
  test_dataset.flags.writeable = False
  valid_dataset.flags.writeable = False

  print ("Train set")
  for instance in train_dataset:
    print( hash(instance.data))

  print ("Test set")  
  for instance in test_dataset:
    print( hash(instance.data))

  print ("Valid set")  
  for instance in valid_dataset:
    print( hash(instance.data))    


imprimirInstancias()
##findInstInSet2(valid_dataset, train_dataset, 0)
##findInstInSet2(valid_dataset, test_dataset, 1)
##
##findInstInSet2(test_dataset, train_dataset, 2)
  

# ---
# Problem 5
# ---------
# 
# By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test samples.
# 
# Optional questions:
# - What about near duplicates between datasets? (images that are almost identical)
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---



# ---
# Problem 6
# ---------
# 
# Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
# 
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
# 
# Optional question: train an off-the-shelf model on all the data!
# 
# ---
