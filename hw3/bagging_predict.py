import numpy as np
import sys
import csv
from keras.utils import np_utils
from keras.models import load_model
import tensorflow as tf
import sys
test = []


testdata = sys.argv[1]
output = sys.argv[2]

with open(testdata,"r") as fin:
    tmp = list(csv.reader(fin))[1:]
    for label,image in tmp:
       pixel = [int(i) for i in image.split(" ")]
       test.append(np.array(pixel).reshape(48,48,1)) 
test = np.array(test, dtype=float)/255
print("test",test.shape)

name = "bagging{}.h5".format(0)
print(name)
model = load_model(name)
y = model.predict(test, batch_size=512, verbose=1)

for i in [1,3,4,5,6,7,8,9]:
    name = "bagging{}.h5".format(i)
    model = load_model("bagging{}.h5".format(i))
    print(name)
    y += model.predict(test, batch_size=512, verbose=1)

out = [['id','label']]
for index,ans in enumerate(y):
    out.append([str(index),np.argmax(ans)])
with open(output,"w") as fout:
    writer = csv.writer(fout)
    writer.writerows(out)

