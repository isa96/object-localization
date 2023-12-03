# object_localization

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project is to localization and predict an object in the image **note: this project only detect cucumber, eggplant, and mushroom due the dataset that I used only contains those object**. I also using flask as a backend to create an API and html as an interface to make a web from it.

# Dataset

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You can get the dataset from [Kaggle - Image Localization Dataset](https://www.kaggle.com/datasets/mbkinaci/image-localization-dataset), The dataset contains object image with jpg format and xml file is contains annotation from the corresponding images. 

![image](https://user-images.githubusercontent.com/91602612/215395129-8cdb0cc4-7df1-49df-9925-587cce783edc.png)

# Notebook

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I built the model in .ipynb file, I used google colab to helped me built the model and this is the explanation about the .ipynb file:
1. I test to plot image with the bounding box, I done this using ```xml.etree.ElementTree``` library to extract xml fit corresponding image, I extract  xmin, ymin, xmax, and ymax from xml file and plot the bounding box around the image using ```cv2.rectangle()``` with xmin, ymin, xmax, and ymax from the xml files, and this is the result

![image](https://user-images.githubusercontent.com/91602612/215397616-6f14fd0d-ed89-4878-b91e-f40e4ac3a818.png)

2. Then I read all xml files to extract label, xmin, ymin, xmax, and ymax from those xml files and append them into list. I encode the categorical value into numerical value **{"cucumber": 0, "eggplant": 1, "mushroom": 2}**, I also read all image files and append the image into list
3. I used ```np.array()``` to convert the lists of image files and outputs (contains label, xmin, ymin, xmax, and ymax)
4. Then I split inputs and outputs array into x_train, x_test, y_train, and y_test, using ```sklearn.model_selection.train_test_split()``` with parameters as follows **test_size = 0.3 and random_state = 42)**
5. Because y_train and y_test has 5 values contains (label, xmin, ymin, xmax, and ymax) I seperate label with other values (coordinate xmin, ymin, xmax, and ymax to build the bounding box) because our model will have 2 outputs (labels and bounding box coordinate) and 1 input (image array).
6. I encode the labels using ```tf.keras.utils.to_categorical()```
7. For the **model** I used pretrained model MobileNetV2 with input_shape = (224,224,3), with 3 classes, weight = 'imagenet' and include_top = False
8. then I added pretrained model into my own layers, I also compile the model with optimizers = Adam(lr=1e-4), loss function has 2 loss for classification is categorical_crossentropy and for bounding box is mse, also in metrics I used 2 metrics, for classification is accuracy and bounding box is mse. Then I fit the model with 50 epochs, and I get this result

![image](https://user-images.githubusercontent.com/91602612/215409573-77dc380f-ddf5-4401-a938-599722ce90af.png)

9. I saved the model to used in API later
10. I test the model to predict image and got predict object localization as follows:

![image](https://user-images.githubusercontent.com/91602612/215409758-11c115b7-5829-49bc-a82a-707e7c308b5d.png)

# Web APP

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For the web app I have:
1. app.py for my backend and build API
2. static folder for save static files like image and predicted image
3. template folder to save html or front end folder

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Here's the result

![WhatsApp Image 2023-01-26 at 01 40 13](https://user-images.githubusercontent.com/91602612/215410363-8dcb4889-5b24-43f9-8b48-a4e91aade7f4.jpeg)




