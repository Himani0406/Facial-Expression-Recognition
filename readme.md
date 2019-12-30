## How to run the code

Download Fer2013 dataset if training of the model is being executed and create a folder with the name data in the main folder and place the fer2013 folder in the data folder- 

    - [Kaggle Fer2013 challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

#For SVM
1. Download model, unzip and place in the SVM folder
	- [Dlib Shape Predictor model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

2. Install dependencies

    ```
    pip install numpy
    pip install argparse
    pip install sklearn
    pip install scikit-image
    pip install pandas
    pip install hyperopt
    pip install dlib
    ```

3. Convert the dataset to extract Face Landmarks and HOG Features

    ```
    python convert_fer2013_to_images_and_landmarks.py
    ```

4. Train the model

    ```
    python train.py --train=yes
    ```

5. Evaluate the model

    ```
    python train.py --evaluate=yes
    ```

6. Train and evaluate [instead of step 5 and 6]

    ```
    python train.py --train=yes --evaluate=yes 
    ```

7. Customize the training parameters:

    Feel free to change the values of the parameters in the `parameters.py` file accordingly.

8. Find the best hyperparameters (using hyperopt):

    ```
    python optimize_parameters.py --max_evals=15
    ```

# For CNN
1. Install Dependencies
	- TensorFlow (latest version) [Installation](https://www.tensorflow.org/install/)
	- OpenCV (python3-version) [Installation](http://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.html)

2. To run the demo, just type:
	```shell
	python3 main.py
	```
Then the program will creat a window to display the scene capture by webcamera. You need press <kbd>SPACE</kbd> key to capture face in current frame and recognize the facial expression.

3. To train models
   Modifying the `MODE`(in `main.py`) from `demo` to `train`  before you start training.
	Then type:
	```shell
	python3 main.py
	```
