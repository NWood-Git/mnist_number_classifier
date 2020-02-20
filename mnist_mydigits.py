import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np   



# Images Drawn in Paint

# image = Image.open('zero_written.jpeg').convert('L') # Got 0
# image = Image.open('one_written.jpeg').convert('L') # Got 1
# image = Image.open('two_written.jpeg').convert('L') # Got 2
# image = Image.open('three_written.jpeg').convert('L') # Got 3
# image = Image.open('four_written.jpeg').convert('L') # Got 4
# image = Image.open('five_written.jpeg').convert('L') # Got 5
# image = Image.open('six_written.jpeg').convert('L') # Got 8 but 6 was close
image = Image.open('six2_written.jpeg').convert('L') #Got 6
# image = Image.open('seven_written.png').convert('L') # Got 7
# image = Image.open('eight_written.jpeg').convert('L') # Got 8
# image = Image.open('nine_written.jpeg').convert('L') # Got 4 -- 9 was a close 2nd
# image = Image.open('nine_written.jpeg').convert('L') # Got 4 -- 9 was 2nd 

# Making the image into pixels so the model can read it
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

# for row in real_pixels:
#     print(row)

arr_num = np.array(real_pixels)
arr_num = np.reshape(arr_num, (1,28,28,1)) # was (28,28,1)

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
# for row in first_train:
#     print(row)

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

prediction = model.predict(arr_num)
print(prediction)
arr_result = np.where(prediction == np.amax(prediction))
# print(arr_result)
# print(arr_result[1])
the_num = arr_result[1][0]
print(f"The number is {the_num}")



#################################################

# The below was used to fit the imported image into model.evaluate.
# It needed to use model predict but it may be helpful code for ref

# print(type(Y_test[0]))
# y_7 = np.array([np.array([0.,0.,0.,0.,0.,0.,0.,1.,0.,0.])])
# # y_7 = np.array(y_7)
# y_7_arr = x = np.zeros([1,10])
# # y_7_arr.add(y_7)
# print(y_7_arr)
#new_score = model.evaluate(arr_gpvalf,y_7, verbose = 1)
#print(new_score)

# The below code did not import the image correctly - may be useful in the future

# #6 made in paint
# image = Image.open("six_written.jpg").convert('L')  # convert image to 8-bit grayscale
# #image = Image.open("six_written.jpg")
# new_image = image.resize((28,28))
# # new_image.show()
# pix_val = list(new_image.getdata())
# pix_val = [abs(x-255) for x in pix_val] #inverted color b/c white background vs black training

# #adding in our own handwritten digits to test the nn###################
# #7 - bad image
# image = Image.open("seven_written.jpg").convert('L')  # convert image to 8-bit grayscale
# new_image = image.resize((28,28))
# new_image.show()
# # print(new_image)
# pixval = list(new_image.getdata())
# gpvalf = [x/255 for x in pixval]
# arr_gpvalf = np.array(gpvalf)
# arr_gpvalf = np.reshape(arr_gpvalf, (1,28,28,1)) # was (28,28,1)