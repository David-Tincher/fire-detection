import time, os, shutil, random, glob, warnings
import tensorflow as tf
from functions import plot_confusion_matrix, plotImages
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix

# Measure the time taken
start_time = time.time()
warnings.simplefilter(action='ignore', category=FutureWarning)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Doesn't work on native Windows with tensorflow > V 2.10

# Organize data into train, valid, test dirs
# ---------------------------------------------------------------------------------------------------
os.chdir('data/fire-vs-non_fire')

# Check if the directory is already exists first
if os.path.isdir('train/fire') is False:
    os.makedirs('train/fire')
    os.makedirs('train/non_fire')
    os.makedirs('valid/fire')
    os.makedirs('valid/non_fire')
    os.makedirs('test/fire')
    os.makedirs('test/non_fire')

    # There are 480 images. fire(240) & non_fire(240)
    # Train : Valid : Test = 8 : 2 : 2
    # Train: 320 (160, 160)
    # Valid: 80 (40, 40)
    # Test: 80 (40, 40)
    for c in random.sample(glob.glob('fire*'), 160):
        shutil.move(c, 'train/fire')
    for c in random.sample(glob.glob('non_fire*'), 160):
        shutil.move(c, 'train/non_fire')
    for c in random.sample(glob.glob('fire*'), 40):
        shutil.move(c, 'valid/fire')
    for c in random.sample(glob.glob('non_fire*'), 40):
        shutil.move(c, 'valid/non_fire')
    for c in random.sample(glob.glob('fire*'), 40):
        shutil.move(c, 'test/fire')
    for c in random.sample(glob.glob('non_fire*'), 40):
        shutil.move(c, 'test/non_fire')

os.chdir('../../')

# Path to the different dataset directories
# ---------------------------------------------------------------------------------------------------
train_path = 'data/fire-vs-non_fire/train'
valid_path = 'data/fire-vs-non_fire/valid'
test_path = 'data/fire-vs-non_fire/test'

# Directory iterator
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224),  batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224),  batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224, 224),  batch_size=10, shuffle=False)

# Checks if datasets are accurate
assert train_batches.n == 320
assert valid_batches.n == 80
assert test_batches.n == 80
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

# Displays some images in train directory after they are pre-processed
# ---------------------------------------------------------------------------------------------------
imgs, labels = next(train_batches)
print(labels)  
plotImages(imgs)

# Fine tuning our CNN to work with our dataset and starting the training
# ---------------------------------------------------------------------------------------------------
mobile = MobileNetV2()
# mobile.summary()   # This line was to show the layers of our CNN

x = mobile.layers[0].output
output = Flatten()(x)
output = Dense(units=2, activation = 'sigmoid')(output)
model = Model(inputs=mobile.input, outputs = output)
tf.keras.optimizers.Adam()
model.compile(optimizer = Adam(learning_rate=0.0001), loss = 'binary_crossentropy', metrics =['accuracy'])
model.fit(train_batches, validation_data=valid_batches, epochs=10, verbose = 2)

# Preparing data to be tested and then making a confusion matrix with the results
# ---------------------------------------------------------------------------------------------------
test_labels = test_batches.classes
predicitons = model.predict(x=test_batches, verbose =0)
cm = confusion_matrix(y_true = test_labels, y_pred=predicitons.argmax(axis=1))
test_batches.class_indices
cm_plot_labels = ['fire', 'non-fire']

# Outputting total time, confusion matrix, and accuracies 
# ---------------------------------------------------------------------------------------------------
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))
plot_confusion_matrix(cm=cm, classes =cm_plot_labels, title = 'Fire Detection')