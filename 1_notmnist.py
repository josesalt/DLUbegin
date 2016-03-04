
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

def showImagePlotNdArray(img2Plot, texto):
  """Plot values from a ndarray."""  
  plt.imshow(img2Plot)
  plt.figtext(0.3,0.025,texto)
  plt.show()  

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
  x = np.vstack((ii,y[ii]))  #.T
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

# Compare the images before shuffling and after  
#  for i in range(1):
#    x = randint(0,len(labels))    
#    showImagePlotNdArray(dataset[permutation[x]],"LABEL before randomize: {0} After Randomize".format(labels[permutation[x]]))
#    showImagePlotNdArray(shuffled_dataset[x],"LABEL AFTER randomize: {0} After Randomize".format(shuffled_labels[x]))

  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)

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


##NUEVO INTENTO TOMADO DE mpacer

import hashlib
import time
from itertools import combinations

def methodMpacer(train, test, valid):
  start = time.time()

  same_set = []
  d_sets = [train, test, valid]
  d_sets_names = ["train_dataset", "test_dataset", "valid_dataset"]
  rep_counts = {}

  for i,x in enumerate(d_sets):
      same_set.append(set([hashlib.sha1(image_array).hexdigest() for image_array in x]))

#      X es el set completo, same_set es el set generado de hash por lo que no tiene valores duplicados
#      print ("orig_len: {0} uniq_len: {1} repeat_count: {2}".format(len(x), len(same_set[i]),len(x)-len(same_set[i])))

      rep_counts[d_sets_names[i]]= {"orig_len": len(x),
                                    "uniq_len": len(same_set[i]),
                                    "repeat_count" : len(x)-len(same_set[i])}
    
# Genera combinaciones posibles entre los conjuntos, los cuales ya son valores hash por el proceso anterior    
  set_pairs = combinations(d_sets,2)
#  print (list(set_pairs))

# Genera combinaciones posibles entre los nombres de los conjuntos
  set_pair_names = combinations(d_sets_names,2)
#  print (list(set_pair_names))
  
  contador = 0
  for i, ((first_set_name, second_set_name),(first_set, second_set)) in enumerate(zip(set_pair_names,set_pairs)):

#     total_set_lengths: Suma del total de instancias en los conjuntos
      total_set_lengths = rep_counts[first_set_name]["orig_len"] + rep_counts[second_set_name]["orig_len"]
      # print (" total_set_lengths: {0}  = rep_counts[first_set_name]['orig_len']: {1} + rep_counts[second_set_name]['orig_len'] {2}".format(
      #           total_set_lengths, rep_counts[first_set_name]["orig_len"], rep_counts[second_set_name]["orig_len"]))

#     total_unique_lens: Suma del total de instancias unicas en base a su valor hash en cada uno de los conjuntos      
      total_unique_lens = rep_counts[first_set_name]["uniq_len"] + rep_counts[second_set_name]["uniq_len"]
      # print (" total_unique_lens: {0}  = rep_counts[first_set_name]['uniq_len']: {1} + rep_counts[second_set_name]['uniq_len'] {2}".format(
      #           total_unique_lens, rep_counts[first_set_name]["uniq_len"], rep_counts[second_set_name]["uniq_len"]))

#     final_unique_lens: Suma del total de instancias unicas de la conjunción de los 2 conjuntos (vstack)
#     pero como se forma un conjunto de valores hash, las ocurrencias son únicas
      # set_NUEVO = set([hashlib.sha1(image_array).hexdigest() 
      #                              for image_array in np.vstack((first_set,second_set))])
      # print (len(set_NUEVO))

      final_unique_lens = len(set([hashlib.sha1(image_array).hexdigest() 
                                   for image_array in np.vstack((first_set,second_set))]))
      # print (" final_unique_lens: {0}".format(final_unique_lens))
 
 # "repeat_count": total_unique_lens - final_unique_lens obtiene la diferencia entre los valores únicos encontrados en cada conjunto respecto a todos
 # y los valores únicos resultado de la conjunción de ambos conjuntos
      rep_counts[first_set_name + " " + second_set_name] = {"total_set_lengths": total_set_lengths,
                                                          "total_unique_lens": total_unique_lens,
                                                          "repeat_count": total_unique_lens - final_unique_lens}
  
  for x in [[key,val["repeat_count"]]for key,val in rep_counts.items()]:
      print(x)
  print(time.time()-start,"s")

  return same_set

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

(samples, width, height) = train_dataset.shape
X = np.reshape(train_dataset,(samples,width*height))

(samples, width, height) = test_dataset.shape
Y = np.reshape(test_dataset,(samples,width*height))

(samples, width, height) = valid_dataset.shape
V = np.reshape(valid_dataset,(samples,width*height))

logRegr = LogisticRegression()

def pruebas(cantidadInstancias): 
  modelo = logRegr.fit(X[0:cantidadInstancias],train_labels[0:cantidadInstancias])
  print ("Resultado entrenamiento con {0:d} instancias {1:.2%}".format (cantidadInstancias, 
          modelo.score(X[0:cantidadInstancias],train_labels[0:cantidadInstancias])))
  print ("Resultado validacion con {0:d} instancias {1:.2%}".format (cantidadInstancias, 
          modelo.score(V[0:cantidadInstancias],valid_labels[0:cantidadInstancias])))
  print ("Resultado prueba con {0:d} instancias {1:.2%} \n".format (cantidadInstancias, 
          modelo.score(Y[0:cantidadInstancias],test_labels[0:cantidadInstancias])))
  
# modelo = logRegr.fit(X,train_labels)
# print ("Resultado entrenamiento con {0:d} instancias {1:.2%}".format (train_labels.size, modelo.score(X,train_labels)))
# print ("Resultado validacion con {0:d} instancias {1:.2%}".format (valid_labels.size, modelo.score(V,valid_labels)))
# print ("Resultado prueba con {0:d} instancias {1:.2%} \n".format (test_labels.size, modelo.score(Y,test_labels)))

######PROCESO PARA SANITANIZAR LOS CONJUNTOS

# Genera tablas de valores hash de los registros repetidos entre el conjunto de entren-prueba y validación-(entren y prueba)
def generateIntersections(train, test, valid):
  same_set = []
  d_sets = [train, test, valid]
  d_sets_names = ["train_dataset", "test_dataset", "valid_dataset"]
  rep_counts = {}

  for i,x in enumerate(d_sets):
      same_set.append(set([hashlib.sha1(image_array).hexdigest() for image_array in x]))

  interTrainTest = same_set[0].intersection(same_set[1])    
  interValidRest = same_set[0].intersection(same_set[2]) 
  interValidRest = interValidRest.union(same_set[1].intersection(same_set[2]) )

  return interTrainTest, interValidRest

dupTrainTest, dupValidRest = generateIntersections(train_dataset, test_dataset, valid_dataset)

# Recibe the dataset to sanitize, the base of the duplicated instances, and the information about the size of the complete set
def generateSanitaizedVersion(data_folders, base, labelsSet, baseRepet, counter, alreadyInBase):
  """Replace the duplicated instances for others in the original database."""
  from collections import deque
  limits = np.zeros(11, np.dtype(np.int16))

  root = os.path.join(os.curdir,data_folders)
#  print (root)

  baseCompl = deque()

  j=0
  acumul=0

  def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in xrange(ord(c1), ord(c2)+1):
      yield chr(c)    
  
  for pickleFile in char_range('A', 'J'):
    pickleFile= pickleFile + '.pickle'
#    print (pickleFile)
    
    pickle_file = os.path.join(root, pickleFile)
    ax = pickle.load(open(pickle_file,  'rb')) 
    for img in ax:
      baseCompl.append(img)
      acumul+=1
    limits[j+1]=acumul
    j+=1

  counter = counter/j
  counters = [counter, counter,counter,counter,counter,counter,counter,counter,counter,counter]

#  print ("Limites: {0}".format(limits))
#  print ("Contadores: {0}".format(counters))
 
 # count_test = 0
  
  for count, image in enumerate(base):    
    if hashlib.sha1(image).hexdigest() in baseRepet:
      label4Subs = labelsSet[count]
#      print ("labels[count]: {0} counters[labels[count]]:{1} limits[labels[count]]:{2}".format(label4Subs,counters[label4Subs],limits[label4Subs]))
      rep = True
      while (rep):        
        if (hashlib.sha1(baseCompl[counters[label4Subs]+limits[label4Subs]]).hexdigest() not in baseRepet and 
            hashlib.sha1(baseCompl[counters[label4Subs]+limits[label4Subs]]).hexdigest() not in alreadyInBase):          
          rep = False

#          if count_test < 10:
#            showImagePlotNdArray(image, "Before ")
          base[count] = baseCompl[counters[label4Subs]+limits[label4Subs]]

#          if count_test < 10:
#            print ("labels[count]: {0} counters[labels[count]]:{1} limits[labels[count]]:{2}".format(label4Subs,counters[label4Subs],limits[label4Subs]))          
#            showImagePlotNdArray(image,"After")
#            count_test+= 1
#          counters[label4Subs]+=1
        else: counters[label4Subs]+=1

        # print ("Limites: {0}".format(limits))
        # print ("Contadores: {0}".format(counters))

  return base

#print ("SETS Size train:{0}, Size test:{1}, Size valid:{2}".format(len(train_dataset), len(test_dataset), len(valid_dataset)))

same_set = methodMpacer(train_dataset, test_dataset, valid_dataset)
#TESTS BEFORE SANITIZATION
#pruebas(50)
#pruebas(100)
#pruebas(1000)
pruebas(20000)

test_dataset = generateSanitaizedVersion("notMNIST_small",test_dataset, test_labels, dupTrainTest, test_size, same_set)
valid_dataset = generateSanitaizedVersion("notMNIST_large",valid_dataset, valid_labels, dupValidRest, train_size+valid_size, same_set)

methodMpacer(train_dataset, test_dataset, valid_dataset)
#TESTS AFTER SANITIZATION
#pruebas(50)
#pruebas(100)
#pruebas(1000)
pruebas(20000)