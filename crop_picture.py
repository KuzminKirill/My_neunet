import string

from PIL import Image, ImageFilter
#from tesserocr import PyTessBaseAPI, RIL


def to_str():
    z = 100
    image = Image.open("picture.jpg")
    with PyTessBaseAPI() as api:
        api.SetImage(image)
        boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        print('Found {} textline image components.'.format(len(boxes)))
        for i, (im, box, _, _) in enumerate(boxes):
            # im is a PIL image object
            # box is a dict with x, y, w and h keys
            api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
            ocrResult = api.GetUTF8Text()
            conf = api.MeanTextConf()
            n = (u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, ""confidence: {1}, text: {2}")
            print(n.format(i, conf, ocrResult, **box))
            print(len(ocrResult))
            print(len(ocrResult.rstrip()))

            x1 = box['x']
            y1 = box['y']
            x2 = box['x'] + box['w']
            y2 = box['y'] + box['h']

            img2 = image.convert('RGB')
            img3 = img2.crop((x1, y1, x2, y2))
            path = "img" + str(z) + ".jpg"
            print("saving in " + path)
            img3.save(path)
            #to_char(path)
            print("saved")
            z += 1





#print(len(ocrResult.rstrip()))

def to_string(path):
    result = []
    image = Image.open(path)
    w, h = image.size
    x = 0
    y = 0
    strok = h // 20
    for i in range(strok):
        img3 = image.crop((x,y, w, y + 20))
        #img3.save(path + str(i) + ".jpg")
        result.append(to_char(img3))
        y = y + 20
    return result

def to_char(path):
    result = []
    #image = Image.open(path)
    image = path
    w, h = image.size
    x = 0
    y = 0
    kol = w // 20
    #print(kol)
    for i in range(kol):
        img3 = image.crop((x, y, x+20, y+20))  # разрезаем на буквы
        #img3.save(path + str(i) + ".jpg")
        result.append(imageprepare(img3))
        x += 20
    return result


#to_str()
#to_char(0)

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    # im = Image.open(argv).convert('L')
    im = argv
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (20, 20), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((20 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((20 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    for j in range(len(tva)):
        if tva[j] >= 0.5:
            tva[j] = 1
        elif tva[j] < 0.5:
            tva[j] = 0

    print(tva)
    return tva

    #s = str(tva).strip('[]')
    #','.join(s)
    #F = open("training_dataset.csv", "a")
    #alphabet = string.ascii_lowercase
    ##print(alphabet)
    ##
    #s = alphabet[count] + ', ' + s + '\n'
    #print(s)
    #F.write(s)
    #print(len(s))

#image_slicer.slice('img2.jpg', len(ocrResult.rstrip()))

#import pytesseract
#from pytesseract import Output
#import cv2
#img = cv2.imread('img2.jpg')
#
#d = pytesseract.image_to_data(img, output_type=Output.DICT)
#n_boxes = len(d['level'])
#for i in range(n_boxes):
#    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#cv2.imshow('img', img)
#cv2.waitKey(0)
