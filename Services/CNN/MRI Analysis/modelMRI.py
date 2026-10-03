from preprocessingCNN import preprocessingCNN
import tensorflow as tf
from tensorflow.keras import layers
data='MRI/Training'

train_data,val_data=preprocessingCNN(data)

model=tf.keras.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(200,200,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(512,activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(5,activation='softmax')
]);

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary();


history=model.fit(train_data,validation_data=val_data,epochs=10)


model.save('MRImodel.h5')