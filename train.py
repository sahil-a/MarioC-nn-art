import numpy as np
import tensorflow as tf


################## LOAD THE DATA ######################
training_files = ["koopa-troopa-beach_trial_1", "koopa-troopa-beach_trial_2", "luigi-raceway_trial_1", "luigi-raceway_trial_2", "mario-raceway_trial_1", "mario-raceway_trial_2", "moo-moo-farm_trial_1", "moo-moo-farm_trial_2"]

all_pic_locs = []
all_vals = []

for training_file in training_files:
    data = np.genfromtxt(f'data/data/{training_file}', delimiter=',', dtype=None)
    pic_locations = []
    vals = []
    for i in range(data.shape[0]):
        (b, val) = data[i]
        vals.append(val)
        pic_locations.append(str(b.decode('UTF-8')))
    all_pic_locs += pic_locations
    all_vals += vals

labels = tf.constant(all_vals)
filenames = tf.constant(all_pic_locs)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

def file_to_image(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    # normalize pixel values to be between 0 and 1
    image /= 255.0
    return image, label

dataset = dataset.map(file_to_image)
# batch size of 128
dataset = dataset.batch(128)


print("done loading images!")


################## CREATE THE MODEL ######################

# see https://www.tensorflow.org/tutorials/images/cnn

from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(1240, 900, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# we only want one output
model.add(layers.Dense(1))

model.summary()

# we use mse loss
model.compile(optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mse'])

# should print loss for each of 10 epochs
history = model.fit(dataset, epochs=10)

