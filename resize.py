from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=2048,height=2048):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
num = 0
for jpgfile in glob.glob("./pic_4096/training/*.jpg"):
    convertjpg(jpgfile,"./training/")
    num += 1
    print(num)