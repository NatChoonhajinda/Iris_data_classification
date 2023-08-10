# Iris_data_classification
 using tensorflow and flask to deploy iris data classification api

# Data
use 
![image](https://github.com/NatChoonhajinda/Iris_data_classification/assets/98221086/fdd31105-99c2-46cf-833c-6bab193718f9)

# tensorflow model
```
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(4, activation='relu'),
tf.keras.layers.Dense(20, activation='relu'),
tf.keras.layers.Dense(88, activation='relu'),
tf.keras.layers.Dense(20, activation='relu'),
tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
                    optimizer = tf.keras.optimizers.Adam(),
                      metrics = ["accuracy"]
                      )


history = model.fit(X_train,
                    y_train,
                    epochs=200,
                    validation_data=(X_test, y_test) )
```

# Evaluate 
```
plt.plot(pd.DataFrame(history.history)['loss'],label = "loss")
plt.plot(pd.DataFrame(history.history)['accuracy'],label = "accuracy")
plt.legend()
plt.figure()
plt.plot(pd.DataFrame(history.history)['val_loss'],label = "val_loss")
plt.plot(pd.DataFrame(history.history)['val_accuracy'],label = "val_accuracy")
plt.legend()
```
![image](https://github.com/NatChoonhajinda/Iris_data_classification/assets/98221086/8aa3bcbd-e694-42b9-b76f-19677d174748)
