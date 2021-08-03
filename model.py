# Import any ML library here (eg torch, keras, tensorflow)
# Start Editing
import torch
# End Editing

import argparse
import random
import numpy as np
from dataLoader import Loader
import os
import cv2

# (Optional) If you want to define any custom module (eg a custom pytorch module), this is the place to do so
# Start Editing
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# End Editing

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
         #Defines the first convolutional layer we will use, because the image input channel is 1, and the first parameter is 1
         #The output channel is 10, kernel_size is the size of the convolution kernel, and the definition here is 5x5
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        #Understand the above definition, you can definitely understand the following
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        #Define another pooling layer
        self.pooling = torch.nn.MaxPool2d(2)
        #Finally is our linear layer for classification
        self.fc = torch.nn.Linear(320, 10)

    #The following is the calculation process
    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0) #The 0 here is the first parameter of x size, which automatically obtains the batch size
        #Input x goes through a convolutional layer, then goes through a pooling layer, and finally uses relu for activation
        x = F.relu(self.pooling(self.conv1(x)))
        #Re-experience the above process
        x = F.relu(self.pooling(self.conv2(x)))
        #To give us the last fully connected linear layer
        #We want to turn a two-dimensional image (actually processed here) 20x4x4 tensor into one-dimensional
        x = x.view(batch_size, -1) # flatten
        #After the linear layer, determine the probability of each number from 0 to 9
        x = self.fc(x)
        return x


# This is the class for training our model
class Trainer:
    def __init__(self):

        # Seed the RNG's
        # This is the point where you seed your ML library, eg torch.manual_seed(12345)
        # Start Editing
        np.random.seed(12345)
        random.seed(12345)
        torch.manual_seed(12345)
        # End Editing

        # Set hyperparameters. Fiddle around with the hyperparameters as different ones can give you better results
        # (Optional) Figure out a way to do grid search on the hyperparameters to find the optimal set
        # Start Editing
        self.batch_size = 16 # Batch Size
        self.num_epochs = 20 # Number of Epochs to train for
        self.lr = 0.0001       # Learning rate
        # End Editing

        # Init the model, loss, optimizer etc
        # This is the place where you define your model (the neural net architecture)
        # Experiment with different models
        # For beginners, I suggest a simple neural network with a hidden layer of size 32 (and an output layer of size 10 of course)
        # Don't forget the activation function after the hidden layer (I suggest sigmoid activation for beginners)
        # Also set an appropriate loss function. For beginners I suggest the Cross Entropy Loss
        # Also set an appropriate optimizer. For beginners go with gradient descent (SGD), but others can play around with Adam, AdaGrad and you can even try a scheduler for the learning rate
        # Start Editing
        self.model=CNN()
        self.loss = nn.CrossEntropyLoss ()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # End Editing

    def load_data(self):
        # Load Data
        self.loader = Loader()

        # Change Data into representation favored by ML library (eg torch.Tensor for pytorch)
        # This is the place you can reshape your data (eg for CNN's you will want each data point as 28x28 tensor and not 784 vector)
        # Start Editing
        self.train_data = torch.tensor(self.loader.train_data)
        self.test_data = torch.tensor(self.loader.test_data)
        self.train_labels = torch.tensor(self.loader.train_labels)
        self.test_labels = torch.tensor(self.loader.test_labels)
        # End Editing

    def save_model(self):
        # Save the model parameters into the file 'assets/model'
        # eg. For pytorch, torch.save(self.model.state_dict(), 'assets/model')
        # Start Editing
        torch.save(self.model.state_dict(),"assets/model")
        # End Editing

    def load_model(self):
        # Load the model parameters from the file 'assets/model'
        if os.path.exists('assets/model'):
        # eg. For pytorch, self.model.load_state_dict(torch.load('assets/model'))
            self.model.load_state_dict(torch.load("assets/model"))
        else:
            raise Exception('Model not trained')

    def train(self):
        if not self.model:
            return

        print("Training...")
        for epoch in range(self.num_epochs):
            train_loss = self.run_epoch()

            # For beginners, you can leave this alone as it is
            # For others, you can try out splitting the train data into train + val data, and use the validation loss to determine whether to save the model or not
            # Start Editing
            self.save_model()
            # End Editing

            print(f'Epoch #{epoch+1} trained')
            print(f"Train loss: {train_loss:.3f}")
            self.test()

    def test(self):
        if not self.model:
            return 0

        print('Running test...')
        # Initialize running loss
        running_loss = 0.0

        # Start Editing
        for x in self.model.parameters():
            x.requires_grad=False
        # Set the ML library to freeze the parameter training

        i = 0 # Number of batches
        correct = 0 # Number of correct predictions
        for batch in range(0, self.test_data.shape[0], self.batch_size):
            batch_X = self.test_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
            batch_Y = self.test_labels[batch: batch+self.batch_size].type(torch.LongTensor) # shape [batch_size,]

            # Find the prediction
            prediction=self.model(batch_X.float())
            # Find the loss
            loss=self.loss(prediction,batch_Y).item()
            # Find the number of correct predictions and update correct
            correct+=(prediction.argmax(1)==batch_Y).type(torch.float).sum().item()
            # Update running_loss
            running_loss+=loss
            i += 1

        # End Editing

        print(f'Test loss: {(running_loss/i):.3f}')
        print(f'Test accuracy: {(correct*100/self.test_data.shape[0]):.2f}%')

        return correct/self.test_data.shape[0]

    def run_epoch(self):
        # Initialize running loss
        running_loss = 0.0

        # Start Editing
        for x in self.model.parameters():
            x.requires_grad=True
        # Set the ML library to enable the parameter training

        # Shuffle the data (make sure to shuffle the train data in the same permutation as the train labels)

        i = 0 # Number of batches
        for batch in range(0, self.train_data.shape[0], self.batch_size):
            batch_X = self.train_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
            batch_Y = self.train_labels[batch: batch+self.batch_size].type(torch.LongTensor) # shape [batch_size,]

            # Zero out the grads for the optimizer
            self.optimizer.zero_grad()
            # Find the predictions
            pred=self.model(batch_X.float())
            # Find the loss
            loss=self.loss(pred,batch_Y)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            # Update the running loss
            running_loss+=loss
            i += 1

        # End Editing

        return running_loss / i

    def predict(self, image):
        prediction = 0
        if not self.model:
            return prediction

        # Start Editing
        image=torch.Tensor([[image]])
        # Change image into representation favored by ML library (eg torch.Tensor for pytorch)
        # This is the place you can reshape your data (eg for CNN's you will want image as 28x28 tensor and not 784 vector)
        # Don't forget to normalize the data (eg. divide by 255 to bring the data into the range of 0-1)
        # Predict the digit value using the model
        prediction=self.model(image)
        # End Editing
        return torch.argmax(prediction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('-train', action='store_true', help='Train the model')
    parser.add_argument('-test', action='store_true', help='Test the trained model')
    parser.add_argument('-preview', action='store_true', help='Show a preview of the loaded test images and their corresponding labels')
    parser.add_argument('-predict', action='store_true', help='Make a prediction on a randomly selected test image')

    options = parser.parse_args()
    t = Trainer()
    if options.train:
        t.load_data()
        t.train()
        t.test()
    if options.test:
        t.load_data()
        t.load_model()
        t.test()
    if options.preview:
        t.load_data()
        t.loader.preview()
    if options.predict:
        t.load_data()
        try:
            t.load_model()
        except:
            print("Not able to load")
        temp=0
        for i in range(1000):
            if t.predict(t.loader.test_data[i][0])!=t.loader.test_labels[i]:
                print(f'Predicted: {t.predict(t.loader.test_data[i][0])}')
                print(f'Actual: {t.loader.test_labels[i]}')
                temp+=1
                print(temp)
                image = t.loader.test_data[i][0].reshape((28,28))
                image = cv2.resize(image, (0,0), fx=16, fy=16)
                cv2.imshow('Digit', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
