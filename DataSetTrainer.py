
# Import os for file path
import os

# Import OpenCV2 for image processing
import cv2

# Import numpy for matrix calculation
import numpy as np

# Import Python Image Library (PIL)
from PIL import Image

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Setting the  file path for the Data Set
path = 'C:\\Users\\Senura Fernando\\PycharmProjects\\FaceRecognition\\DataSet'

# Create method to get the images and label data
def getImagesWithID(path):

    # Get all file path
    # Listing all the Images using (os.listdir)
    #
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    # Initialize empty face sample
    faces = []

    # Initialize empty id
    IDs = []

    # Loop all the images
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        faceImg = Image.open(imagePath).convert('L')

        # Get the image and converting it to a numpy array
        faceNp = np.array(faceImg,'uint8')

        # Get the image id
        ID = int(os.path.split(imagePath) [-1].split('.') [1])

        # Get the face from the training images
        faces.append(faceNp)

        # Printing the IDs
        print ("ID")

        # Add the ID to IDs
        IDs.append(ID)

        cv2.imshow("training",faceNp)

        cv2.waitKey(10)

    # Pass the face array and IDs array
    return np.array(IDs), faces

# Get the faces and IDs
Ids, faces = getImagesWithID(path)

# Train the model using the faces and IDs
recognizer.train(faces,Ids)

# Save the model into trainer.yml
recognizer.save('Trainer/Trainer.yml')

# Clear All
cv2.destroyAllWindows()