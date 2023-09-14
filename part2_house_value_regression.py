import torch
import torch.nn as nn 
import pickle
import numpy as np
import pandas as pd
import copy
from sklearn import preprocessing, impute, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


class Regressor():

    @staticmethod
    def initialise_weights(layer):
        # For Linear Layer
        if type(layer) == nn.Linear:
            # Apply the normal distribution to your weight
            nn.init.normal_(layer.weight)
            # Fill all the bias with 1 
            layer.bias.data.fill_(1)

    def __init__(self, x, nb_epoch = 500, neurons = [4,4], learning_rate = 0.001, batch_size = 512):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.x = x
        # Implements normalisation of data using MinMax scaler
        self.x_scaler = preprocessing.MinMaxScaler()
        self.y_scaler = preprocessing.MinMaxScaler()

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.batch_size = batch_size

        #Initialise number of neurons and learning rate
        self.neurons = neurons
        self.learning_rate = learning_rate

        # Appending layers using torch.nn
        self._layers = [] 
        temp_size = self.input_size
        for neuron in neurons: 
            self._layers.append(nn.Linear(temp_size, neuron))
            self._layers.append(nn.ReLU())
            temp_size = neuron

        #Append last linear layer and sigmoid activation to predict
        self._layers.append(nn.Linear(self.neurons[-1], 1))
        self._layers.append(nn.Sigmoid())

        # Creates a sequential of layers, obtained from https://stackoverflow.com/questions/46141690/how-do-i-write-a-pytorch-sequential-model
        self.net = nn.Sequential(*self._layers)
        self.net.apply(self.initialise_weights) # Applies the weights to the linear layers of the network
        self.net.double()

        #Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        self.loss_tracker = float('inf')
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Imputer replaces all missing values in data with its mean. 
        imputer = impute.SimpleImputer(missing_values=np.nan, strategy = 'mean')
        
        # Creates LabelBinarizer object
        lb = preprocessing.LabelBinarizer()
        lb.classes_ = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
        # Converts "ocean_proximity" label to one hot encoding
        one_hot = x.copy()
        one_hot = lb.transform(one_hot['ocean_proximity'])
        # Removes the "ocean_proximity" column from original dataset
        x = x.drop('ocean_proximity', axis=1)

        # Fill in all the missing values with the mean of total bedrooms
        x = imputer.fit_transform(x)

        # Normalises the dataset
        if training is True:
            self.x_scaler.fit(x)

        x = self.x_scaler.transform(x)

        # Append the one hot encoding to the normalised array
        x = np.concatenate((x, one_hot), axis = 1)
        
        # Convert numpy array to tensor
        x = torch.from_numpy(x)

        if y is not None:
            # Normalise the y dataset
            if training is True:
                self.y_scaler.fit(y)

            y = self.y_scaler.transform(y)
            # Converts numpy to tensor
            y = torch.from_numpy(y)

        # Return preprocessed x and y, return None for y if it was None
        return x, (y if isinstance(y, torch.Tensor) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, x_val = None, y_val = None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        for i in range(self.nb_epoch):
            #Shuffle the indices
            indices = torch.randperm(len(X))
            X = X[indices]
            Y = Y[indices]
            # Perform batch testing and updating gradients based on losses
            for j in range(0, len(X), self.batch_size):
                batch_X = X[j*self.batch_size:(j+1)*self.batch_size]
                batch_Y = Y[j*self.batch_size:(j+1)*self.batch_size]
                self.optimizer.zero_grad()
                output = self.net(batch_X)
                loss = self.criterion(output,batch_Y)
                loss.backward()
                self.optimizer.step()       

            # If we have validation sets
            if x_val is not None and y_val is not None:
                # Calculate loss of validation set 
                loss = self.score(x_val, y_val)
                # If loss less than current loss tracker, store it
                if loss < self.loss_tracker:
                    # The ith iteration
                    self.counter = i
                    self.loss_tracker = loss
                    # Keep a copy of current "best" network
                    regression = copy.deepcopy(self)
                else:
                    # If the loss has been lesser for 10 consecutive times and its less than 87000, return the best regression
                    if i - self.counter >= 10 and (self.loss_tracker) < 87000:
                        return regression
            
        return self
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
            
        X, _ = self._preprocessor(x, training = False) # Do not forget
        output = self.net(X).detach().numpy()
        return self.y_scaler.inverse_transform(output)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        # The inverse transform function tells me to do this
        output = self.net(X).detach().numpy()

        output_num = self.y_scaler.inverse_transform(output)

        # Change our output to house prices
        output_y = self.y_scaler.inverse_transform(Y)

        # Compute the mean squared error for our model's prediction and actual output
        error_mse = metrics.mean_squared_error(output_num, output_y, squared = False)
        return error_mse

    # Copied these needed functions for GridSearchCV https://scikit-learn.org/stable/developers/develop.html
    def get_params(self, deep=True):
        return {
            'x': self.x,
            'nb_epoch': self.nb_epoch,
            'neurons': self.neurons,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################



def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")



def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x, y): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    #80% Training, 20% Val + Test
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size = 0.2, random_state=111) 

    # 10% Val, 10% Test
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size= 0.5, random_state=111)

    search_space = {
        "nb_epochs" : [500],
        "neurons" : [[1,4],[2,4],[3,4],[4,4],[5,4],[6,4],[7,4],[8,4],[4,4,4],[4,5,4],[4,6,4]],
        "learning_rate" : [0.001],
        "batch_size" : [32, 64, 128, 256, 512, 1024, 2048]
    }

    grid_search = GridSearchCV(
        Regressor(x=x_train), 
        param_grid = search_space, # Our hyperparameters to tune
        n_jobs= -1, # Run the tuning in parallel
        scoring=['r2', 'neg_root_mean_squared_error'],
        verbose=3, # Prints all the data possible
        cv=4, # Number of cross validation sets
        refit= 'neg_root_mean_squared_error',
        return_train_score=False)

    grid_search.fit(x_train, y_train, x_val=x_val, y_val=y_val)

    print(grid_search.best_params_)
    print(grid_search.best_score_)
    df = pd.DataFrame(grid_search.cv_results_)
    df.to_csv("file.csv")

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 
    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    #80% Training, 20% Val + Test
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size = 0.2, random_state=111) 

    # 10% Val, 10% Test
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size= 0.5, random_state=111)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, neurons = [2,4], nb_epoch = 500, batch_size = 512)
    regressor.fit(x_train, y_train, x_val = x_val, y_val = y_val)
    save_regressor(regressor)
    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))
    #RegressorHyperParameterSearch(x, y)

if __name__ == "__main__":
    example_main()