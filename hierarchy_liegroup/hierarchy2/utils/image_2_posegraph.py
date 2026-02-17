from PIL import Image
import numpy as np

image = Image.open('./resources/GBP.png')
width, height = image.size
ratio = 1920/width
image = image.resize((int(width * ratio), int(height * ratio)),Image.Resampling.NEAREST)
width, height = image.size
img = np.array(image)
#image.show()

var_x = []
var_y = []

for w in range(0, 1920, 40):
    for h in range(0, height, 40):
        if np.mean(img[h,w][0:3]) < (50):
            var_x.append(w)
            var_y.append(height - h)

with open(r'meaning_of_life_x.txt', 'w') as fp:
    for item in var_x:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

with open(r'meaning_of_life_y.txt', 'w') as fp:
    for item in var_y:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')
