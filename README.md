# Sudoko-solver

Given an image, this code detects whether a sudoku is present and extracted digits from it using openCV library in python. I have implemented a CNN to train on MNIST dataset, my model gives an accuracy of about 98.66 percent database. This model is used to predict the extracted digit. Due to these being printed digits rather than handwritten MNIST images, accuracies are slightly lower and i have manually added 2-3 corrections to the sample sudokus present.

In the assests/sudokus folder there are pictures of solved and unsolved sudokus, theres also a folder called bad which has sudokus which couldnt get detected properly.

To run the code for any sudoku,
1) Install all the requirements by running the command pip -r install requirements.txt.
2) run sudoku.py it as python3 sudoku.py name.jpg, where name.jpg is the name of the sudoku u want to run.
3) Now, if solved, a solved sudoku with name name_solved.png will be saved in the assets/sudokus.

If you want to make changed in the CNN model, you can edit the model.py file and train the model by running the command:
python3 model.py -train.
