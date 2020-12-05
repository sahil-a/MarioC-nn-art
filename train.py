import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

################## LOAD THE DATA ######################

#Hyperparameters
test_split = 0.3
val_split = 0.2
bs = 32

training_files = ["koopa-troopa-beach_trial_1", "koopa-troopa-beach_trial_2", "luigi-raceway_trial_1", "luigi-raceway_trial_2", "mario-raceway_trial_1", "mario-raceway_trial_2", "moo-moo-farm_trial_1", "moo-moo-farm_trial_2"]
# training_files = ["koopa-troopa-beach_trial_1"]
all_pic_locs = []
all_vals = []

for ind, training_file in enumerate(training_files):
    data = np.genfromtxt(f'data/data/{training_file}', delimiter=',', dtype=None)
    pic_locations = []
    vals = []
    for i in range(data.shape[0]):
        (b, val) = data[i]
        vals.append(val)
        pic_locations.append(str(b.decode('UTF-8')))
    all_pic_locs += pic_locations
    all_vals += vals
    print("Loaded file %d" % ind)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(all_pic_locs, all_vals, test_size=test_split)
# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split)

# Converting to tensors
train_filenames = tf.constant(X_train)
train_labels = tf.constant(y_train)

val_filenames = tf.constant(X_val)
val_labels = tf.constant(y_val)

test_filenames = tf.constant(X_test)
test_labels = tf.constant(y_test)

# Convert to datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))

def file_to_image(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    # normalize pixel values to be between 0 and 1
    image /= 255.0
    return image, label

# Code to view examples from dataset
# image1 = tf.image.decode_png(tf.read_file('/usr/src/pycharm-2017.1/bin/pycharm.png'))
# print(image1.shape)
# with tf.Session() as sess:
#     img = sess.run(image1)
#     print(img.shape, img)

# Map file paths to images
train_dataset = train_dataset.map(file_to_image)
test_dataset = test_dataset.map(file_to_image)
val_dataset = val_dataset.map(file_to_image)

# sets batch size
train_dataset = train_dataset.batch(bs)
test_dataset = test_dataset.batch(bs)
val_dataset = val_dataset.batch(bs)

print("_________________________________________________________________")
print("DONE LOADING IMAGES!")
print("_________________________________________________________________")


################## CREATE THE MODEL ######################
# see https://www.tensorflow.org/tutorials/images/cnn

from tensorflow.keras import layers, models

#Set hyperparameters:
lr = 0.01
epochs = 15

#First model - final activation tanh
def create_model_tanh():
    modeltype = "tanh1"
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ['crossentropy']

    # Actual model
    model = models.Sequential()
    model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(1240, 900, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # we only want one output that is squashed between -1 and 1
    model.add(layers.Dense(1, activation='tanh'))
    return model, modeltype, loss, metrics

# Second model - final activation linear
def create_model_linear():
    modeltype = "linear1"
    loss = tf.keras.losses.MeanSquaredError()
    metrics = ['mse']

    # Actual model
    model = models.Sequential()
    model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(1240, 900, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    return model, modeltype, loss, metrics

# Choose which model to train
model, modeltype, loss, metrics = create_model_linear()
checkpoint_filepath = "models/" + modeltype
model.summary()

# checkpoint to save model every epoch based on loss
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)
opt = keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt,
        loss=loss,
        metrics=metrics)

# should print loss for each of 10 epochs
history = model.fit(train_dataset, epochs=epochs, validation_data = val_dataset,
                    callbacks = [model_checkpoint_callback])

# Evaluation on test set
print("_________________________________________________________________")
print("EVALUATION:")
result = model.evaluate(test_dataset)
print(dict(zip(model.metrics_names, result)))
print("_________________________________________________________________")

print("Saving data...")

################## SAVE DATA ON MODEL ######################
f = open("results/model%sdata" % modeltype, "a")
model.summary(print_fn=lambda x: f.write(x + '\n'))
f.write("Final: ")
f.write(str(dict(zip(model.metrics_names, result))))

# list all data in history
print(history.history.keys())
# summarize history for loss
loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
loss.savefig('results/model%sloss.png' % modeltype)

print("DONE!")

