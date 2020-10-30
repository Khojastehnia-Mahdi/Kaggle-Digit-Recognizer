# accuracy vs iterations

TrainAcc = ModelHistory.history['accuracy']
ValAcc = ModelHistory.history['val_accuracy']
TrainLoss = ModelHistory.history['loss']
ValLoss = ModelHistory.history['val_loss']

Nepochs = range(len(TrainAcc))

plt.plot(Nepochs, TrainAcc, 'k', label='Training')
plt.plot(Nepochs, ValAcc, 'r', label='Validation')
plt.title('accuracy - Training & validation')
plt.ylabel('Accuracy[%]')
plt.xlabel('Iteration')
plt.legend(loc=0)
plt.figure()


plt.show()


# scaling the images in the test set
test = X_test/255

# predicting for the test set
results = model.predict(test)

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
