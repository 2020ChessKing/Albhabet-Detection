# Imports :
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.ImageOps

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

# Fetching Data :
X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

# Train ---> Test ---> Split :
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

# Scaling Features : 
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# Fitting Train Data ---> Model :
clf = LogisticRegression(solver='saga', multi_class='multinomial')
clf.fit(X_train_scaled, y_train)

# Accurcy Calculation :
y_pred = clf.predict(X_test_scaled)
accuracy =  accuracy_score(y_test, y_pred)
print("The Accuracy of This Model is ----> ", accuracy)


# Camera Start :
cap = cv2.VideoCapture(0)

while(True):

    # Frame-by-Frame Capture
    try:
        ret, frame = cap.read()

        # FRAME OPERATIONS
        # Grey-scaling :
        gray = cv2.cvtColor(frame, cv2.COLOR_BG2GRAY)

        # Centered-Box
        height, width = gray.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        # Region of Intrest (ROI)
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        # Convert CV2 ---> PIL Image Format :
        im_pil = Image.fromarray(roi)

        # Convert Image to Grayscale 'L' Format :
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scale = np.clip(image_bw_resized_inverted - min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled) / max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)

        print("Predicted Class Is ---> ", test_pred)

        # Display Output Frame :
        cv2.imshow('frame', gray)

        if cv2.waitkey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        pass

# Release Capture :
cap.release()
cv2.destroyAllWindows()