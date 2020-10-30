# CNN (convolutional neural network) model: accuracy = 99.5%

# This model is inspired by the model in https://www.kaggle.com/alifrahman/digit-recognizer-for-beginners-0-9966

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1), padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(rate = 0.4),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (5,5), activation='relu', padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(rate = 0.25),
    tf.keras.layers.Conv2D(128, (4,4), activation='relu', padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (4,4), activation='relu', padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 512, activation='relu'),
    tf.keras.layers.Dropout(rate = 0.7),
    tf.keras.layers.Dense(units = 128, activation='relu'),
    tf.keras.layers.Dropout(rate = 0.7),
    tf.keras.layers.Dense(units = 10, activation='softmax')
])

model.summary()


optimizer = tf.keras.optimizers.Adam()
#optimizer = tf.keras.optimizers.RMSprop()


model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


from tensorflow.keras.callbacks import LearningRateScheduler

# Decreasing the learning rate in each iteration
callbackLR = LearningRateScheduler(lambda x: 1e-3 * (0.97)**x)

# I tried 16, 32 and 128 for the batch size. 
# Then, I observed that 32 is the optimal value for the batch size. 

ModelHistory = model.fit(training_datagen.flow(X_train, y_train, batch_size=32) 
                  , epochs=40, steps_per_epoch=X_train.shape[0]//32,
                    validation_data = validation_datagen.flow(X_valid, y_valid),
                    verbose = 2, callbacks = [callbackLR])

