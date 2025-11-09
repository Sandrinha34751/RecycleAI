# (translated)  model.py
# (translated)  Define, treina e salva o model.
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
from data_utils import FilteredImageSequence

def build_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

def train(train_dir, val_dir, use_filters=True,
          img_size=(128,128), batch_size=32, epochs=20, save_path='recycle_model.h5'):
    train_seq = FilteredImageSequence(train_dir, batch_size=batch_size, size=img_size, use_filters=use_filters)
    val_seq = FilteredImageSequence(val_dir, batch_size=batch_size, size=img_size, use_filters=use_filters, shuffle=False)

    num_classes = len(train_seq.class_indices)

    X0, y0 = train_seq[0]
    input_shape = X0.shape[1:]  # (translated)  (H, W, C)

    model = build_cnn(input_shape, num_classes)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=callbacks)
    
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(train_seq.class_indices, f, indent=2)
    return model, history
