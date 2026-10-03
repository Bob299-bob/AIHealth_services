import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers

def preprocessingCNN(data):
    train_data=image_dataset_from_directory(data,image_size=(200,200),
                                            validation_split=0.2,
                                            shuffle=True,
                                            subset='training',batch_size=100,seed=1)
    val_data=image_dataset_from_directory(data,image_size=(200,200),
                                          validation_split=0.2,
                                          shuffle=False,
                                          subset='validation',batch_size=100,seed=1)
    
    train_data=train_data.map(lambda x,y:(x/255.0,y))
    val_data=val_data.map(lambda x,y:(x/255.0,y))

    return train_data,val_data