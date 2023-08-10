# Iris_data_classification
 using tensorflow and flask to deploy iris data classification api
![image](https://github.com/NatChoonhajinda/Iris_data_classification/assets/98221086/ca34d9f2-f9fb-4d19-83d3-c1ec9f9caada)

# Data

![image](https://github.com/NatChoonhajinda/Iris_data_classification/assets/98221086/65f564e3-88f1-457c-8c47-2643b745a0f8)

# pre-processing
```
dataset = pd.get_dummies(dataset, columns=['species'])
```
![image](https://github.com/NatChoonhajinda/Iris_data_classification/assets/98221086/779a600b-8d49-4f5e-8e5f-0dd1159b9cc9)

# tensorflow model
- The main purpose of this project is deploy the tensorflow model by Flask. 
- i not pay much attention to AI model because it's still good.
- the main concept is use RELU to extract feature and then use SofeMax to classify
  Note
  - Softmax return length of Array equal to how much data you want to classify
  - it's something like this [3.26865047e-05, 8.21033776e-01, 1.78933561e-01]
  - make sure you use "Round" to make it 0 , 1
  - more info "https://www.geeksforgeeks.org/numpy-round_-python/"
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

# FLASK
Load model to Flask and pass Json data by postman
```
from flask import Flask ,request
import tensorflow as tf
import numpy as np
import json
import pandas as pd

loaded_model = tf.keras.models.load_model('my_model')
app = Flask(__name__)

@app.route('/', methods=['POST'])
def return_word():
    #load tf model
    loaded_model = tf.keras.models.load_model('my_model')
    # get request data
    data = request.get_json()
    data_array = pd.DataFrame(data)
    
    
    #predict
    ans = np.round_(loaded_model.predict(data_array))
    json_data  = json.dumps(ans.tolist())
    return json_data

if __name__ == '__main__':
    app.run()


```
# pros
- Flexible
- Good for deploy tensorflow model (with server)
- Automate

# Problem
- Flask doesn't work sometimes with chatgpt code
- miss format Json file that's make AI broke(if Ai doesn't make a good classify. make sure recheck the json u pass in)
- sometimes it's show error like this when use sklearn version but it's fine.
![image](https://github.com/NatChoonhajinda/Iris_data_classification/assets/98221086/131bfeb8-7ad1-4c37-97ce-85b767543c7e)

# personal Problem
- lack of Debug experience
- lack of Back-end experience
