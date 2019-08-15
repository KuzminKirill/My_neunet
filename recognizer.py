#Usage CUDA_DEVICE=0 python3 Network.py
from Network import neuralNetwork
from crop_picture import to_char, to_string
import pycuda.driver as drv


drv.init()
print("%cuda device work " % drv.Device.count())
for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print("Device #%d: %s" % (ordinal, dev.name()))
    print("Compute Capability: %d.%d" % dev.compute_capability())
    print("Total Memory: %s KB" % (dev.total_memory() // 1024))

n = neuralNetwork()
n.load("wih.npy", "who.npy")

#im = Image.open("picture.jpg")

vectors = to_string("picture_times_2.jpg")


print(len(vectors))
print(len(vectors[1]))

symbols = ''

def isit(zero_vect):
    count = 0
    for i in range(len(zero_vect)):
        if zero_vect[i] == 1:
            count += 1
    if count < 11:
        print('Пробел!')
        return True
    else:
        print('не пробел')
        return False

for i in range(len(vectors)):
    z = 0
    symbols = symbols + ' '
    for j in range(len(vectors[i])):
        temp = vectors[i][j]
        if isit(temp):
            symbols = symbols + ' '
        else:
            symbols = symbols + n.query(temp)

print(symbols)








