import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import splitfolders
import tensorboard
import tensorflow as tf
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard
from keras.models import model_from_json
from tensorflow import keras

img_folder = 'Dataset'
splitfolders.ratio(img_folder, output="Data", ratio=(.7, .15, .15), group_prefix=None)

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 10 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'Data/train',  # This is the source directory for training images
        classes = ['no_cow', 'cow'],
        target_size=(180, 180),  # All images will be resized to 180x180
        batch_size=10,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 10 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        'Data/val',  # This is the source directory for validation images
        classes = ['no_cow', 'cow'],
        target_size=(180, 180),  # All images will be resized to 180x180
        batch_size=10,
        # Use binary labels
        class_mode='binary',
        shuffle=False)

# Flow validation images in batches of 10 using valid_datagen generator
test_generator = validation_datagen.flow_from_directory(
        'Data/test',  # This is the source directory for testing images
        classes = ['no_cow', 'cow'],
        target_size=(180, 180),  # All images will be resized to 180x180
        batch_size=1,
        # Use binary labels
        class_mode='binary',
        shuffle=False)

def trainModel():
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(180,180,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) 

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    history = model.fit(train_generator, epochs=20, verbose=1, validation_data = validation_generator)
    model.evaluate(test_generator)

def vgg_model():
    base_model = tf.keras.applications.VGG19(weights="imagenet", include_top=False, input_shape=(180,180,3))  
    for layer in base_model.layers:
        layer.trainable = False

    last_layer = base_model.get_layer('block5_pool')
    last_output = last_layer.output
    x = tf.keras.layers.GlobalMaxPooling2D()(last_output)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)  
    model = tf.keras.Model(base_model.input, x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=20, verbose=1, validation_data = validation_generator)
    model.save('VGG-CNN.h5')
    model.evaluate(test_generator)

def newCustomModel():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(180,180,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=[keras.metrics.Precision(),keras.metrics.Recall(),'accuracy'])

    history = model.fit(train_generator, epochs=20, verbose=1, validation_data = validation_generator)
    model.save('custom-CNN.h5')

def multipleLayerModelLoop():
    dense_layers = [0, 1]
    layer_sizes = [32, 64, 128, 256]
    conv_layers = [1, 2, 3]
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense".format(conv_layer, layer_size, dense_layer)
                print(NAME)
                model = Sequential()

                model.add(Conv2D(64, (3, 3), input_shape=(180,180,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(3, 3)))

                for l in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())
                model.add(Dropout(0.5))

                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))
                    
                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

                history = model.fit(train_generator, epochs=20, verbose=1, validation_data = validation_generator, callbacks=[tensorboard])

######### UNCOMMET ANY ONE FUNCTION CALL - THIS WILL TRAIN MODEL AGAIN #######
#trainModel()
#vgg_model()
#newCustomModel()

model = keras.models.load_model('custom-CNN.h5')
probabilities = model.predict_generator(generator=test_generator)
y_pred = probabilities > 0.5
#display results of model testing
print(y_pred)
loss, precision, recall, accuracy = model.evaluate(test_generator)    
print("Accuracy: "+str(accuracy))
print("F1-Score: "+ str(2 * (precision * recall) / (precision + recall)))
print("Precision: "+str(precision))
print("Recall: "+str(recall))

#generate confusion matrix based on predictions made above
cm = confusion_matrix(test_generator.classes, y_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.show()