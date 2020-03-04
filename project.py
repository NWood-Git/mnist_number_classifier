import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np   
import pickle



(X_train,Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255

Y_train_one_hot = to_categorical(Y_train)
Y_test_one_hot = to_categorical(Y_test)

first_train = X_train[0]

model = Sequential()
model.add(Conv2D(32, kernel_size = (5,5), strides = (1,1), activation = "relu", input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "valid"))
model.add(Conv2D(64, kernel_size = (5,5), strides = (1,1), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "valid"))
model.add(Flatten())
model.add(Dense(300, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train_one_hot, batch_size=100, epochs=3)
score = model.evaluate(X_test, Y_test_one_hot, batch_size=100, verbose = 1)

print("Training and Testing - Done")
image = Image.open('six2_written.jpeg').convert('L') #Got 6

pix_val = []
for x in list(image.getdata()):
    pixel = abs(x-255) / 255
    pix_val.append(pixel)

real_pixels = []
i = 0
while i < len(pix_val):
    cur_row = []
    for j in range(i, i+28):
        cur_row.append(pix_val[j])
    real_pixels.append(cur_row)
    i += 28

arr_num = np.array(real_pixels)
arr_num = np.reshape(arr_num, (1,28,28,1)) 

# saved_model = pickle.dumps(model)
# pickled_model = pickle.loads(saved_model)

# prediction = pickled_model.predict(X_test)
# print(prediction)

prediction = model.predict(arr_num)
arr_result = np.where(prediction == np.amax(prediction))
the_num = arr_result[1][0]
print(f"The number is {the_num}")

model.save("model.h5")