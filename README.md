# Face-detection-Assignment
<h1>Reqirement</h1>
opencv

​ I use cv2 for image io and resize(much faster than skimage), the input image's channel is acutally BGR

Flickr-Faces-HQ Dataset (FFHQ)

Import a datset in image folder
The dataset consists of 52,000 high-quality PNG images at 512×512 resolution and contains considerable variation in terms of age, ethnicity and image background. It also has good coverage of accessories such as eyeglasses, sunglasses, hats, etc. 

<h1>Test</h1>

The output has to be the coordinates of three bounding  boxes - two maximally enclosing the eyes and one enclosing the mouth
run:
python main.py

![image](https://github.com/ajoshi222/Face-detection-Assignment/assets/69758727/0ba1bcdf-e000-45d5-978c-371682f182f2)



You also check live image testing of detection eyes and mouth
python face.py

![image](https://github.com/ajoshi222/Face-detection-Assignment/assets/69758727/d651986d-6ee8-499a-a5a5-72d997cb3b93)

