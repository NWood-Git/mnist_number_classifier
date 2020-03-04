from PIL import Image
import numpy as np   
# import pickle
from keras.models import load_model

model = load_model("model.h5")


def run(image):
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
    return arr_num



# prediction = model.predict(arr_num)
# arr_result = np.where(prediction == np.amax(prediction))
# the_num = arr_result[1][0]
# print(f"The number is {the_num}")

# prediction = model.predict(arr_num)
# arr_result = np.where(prediction == np.amax(prediction))
# the_num = arr_result[1][0]
# print(f"The number is {the_num}

image0 = Image.open('zero_written.jpeg').convert('L')
image1 = Image.open('three_written.jpeg').convert('L')
image2 = Image.open('zero_written.jpeg').convert('L')
image3 = Image.open('four_written.jpeg').convert('L')
image4 = Image.open('two_written.jpeg').convert('L')
image5 = Image.open('zero_written.jpeg').convert('L')
# images = [Image.open('one_written.jpeg').convert('L'), Image.open('five_written.jpeg').convert('L'), Image.open('eight_written.jpeg')]
images = [image0,image1,image2,image3,image4,image5]


result = []
for image in images:
    arr_num = run(image)
    prediction = model.predict(arr_num)
    arr_result = np.where(prediction == np.amax(prediction))
    the_num = arr_result[1][0]
    result.append(the_num)
# print(result)
print(f"{result[0]}{result[1]} / {result[2]}{result[3]} / {result[4]}{result[5]}")