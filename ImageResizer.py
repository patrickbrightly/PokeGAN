import glob
from PIL import Image, UnidentifiedImageError
import os

'''
This will be used to resize all images in one directory and place the resized images in a new directory
'''
class ImageResizer:

    def __init__(self, new_size, mode='RGB', conv_filter=Image.LANCZOS):
            self.new_size = new_size
            self.mode = mode
            self.conv_filter = conv_filter

    def resize(self,in_path,out_path):
        if not os.path.exists(in_path):
            raise Exception("input path is not valid")
        if not os.path.exists(out_path):
            print(out_path,'did not exist, new directory created')
            os.makedirs(out_path)
        files=glob.glob(in_path+'*')
        for old_img in files:
            try:
                img = Image.open(old_img)
                if img.size !=self.new_size:
                    img = img.resize(self.new_size,resample=self.conv_filter)
                if img.mode != self.mode:
                    img.convert(self.mode)
                img.save(os.path.join(out_path,os.path.split(old_img)[-1]))
            except UnidentifiedImageError:
                print(str(old_img), 'could not be converted. Please check filetype')

inpath = './data/sugimori/'
outpath = './data/sugimori/'

resizer = ImageResizer((64,64))
resizer.resize(inpath,outpath)