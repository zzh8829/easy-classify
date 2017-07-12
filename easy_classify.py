import json
import glob
import os
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import cv2

class EasyClassify:

    def __init__(self, data_path):
        self.data_path = data_path
        self.config = json.load(open(data_path + '/config.json'))
        self.model_path = self.data_path + '/model.h5'

        shape = (self.config['shape'][0], self.config['shape'][1], 3)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.config['class']), activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        self.model = model

    def load(self):
        try:
            if os.path.isfile(self.model_path):
                self.model.load_weights(self.model_path)
                print('Model Loaded')
            else:
                print('No Model Found')
        except Exception as e:
            print('Load Failed ', e)

    def train(self, reset=True, tensorboard=False):
        if not reset:
            self.load()

        x = []
        y = []
        for cls,i in enumerate(self.config['class']):
            for img in glob.glob("%s/%s/*.png"%(self.data_path, cls)):
                data = img_to_array(load_img(img))
                x.append(data)
                y.append(i)

        train_gen = ImageDataGenerator(
            rotation_range=3,
            width_shift_range=0.01,
            height_shift_range=0.01,
            shear_range=0.01,
            zoom_range=0.01,
            fill_mode='nearest'
        ).flow_from_directory(self.data_path,
            batch_size=32,
            target_size=self.config['shape'],
            class_mode='categorical',
            save_to_dir='./gen',
            save_prefix='train')

        validation_gen = ImageDataGenerator().flow_from_directory(self.data_path,
            batch_size=32,
            target_size=self.config['shape'],
            class_mode='categorical')

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=1),
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True, verbose=1),
        ]

        if tensorboard:
            callbacks.append(TensorBoard(log_dir='./log/%d'%time.time(), histogram_freq=0,
            write_graph=True, write_images=True))

        self.model.fit_generator(
                train_gen,
                steps_per_epoch=100,
                epochs=100,
                validation_data=validation_gen,
                validation_steps=20,
                callbacks=callbacks)

    def classify(self, img):
        img = cv2.resize(img, tuple(self.config['shape']))
        img = img.reshape([1, self.config['shape'][0], self.config['shape'][1], 3])
        return self.model.predict(img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=string, required=True)
