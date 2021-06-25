import os
import numpy as np
from PIL import Image

#mulitiply image with 0/1 0/255 mask

path='./'  #./img ./mask
i=0
for file,mask in zip(os.listdir(os.path.join(path,"img")),os.listdir(os.path.join(path,"mask"))):
    pic=Image.open(os.path.join(path,"img",file))
    #pic=pic.convert('L')
    mk = Image.open(os.path.join(path, "mask", mask))
    mul=np.array(mk)/255
    final=Image.fromarray(np.array(pic)*mul)
    final=final.convert('L')#for grayscale
    #final=final.convert('RGB') #for RGB
    #final.show() wether show image
    final.save(os.path.join(path,"final",str(i)+".jpg"))
    i=i+1


