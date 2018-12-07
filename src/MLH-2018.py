
# coding: utf-8

# # Jupyter Notebook for the Stock Prediction Data Analysis ( SPDA )

# ## Data Source : https://www.kaggle.com/qks1lver/amex-nyse-nasdaq-stock-histories

# # Development stages
# 
# ### 1. [Dataset formatting and freature extraction](#dataset_formatting_and_extraction)
# ### 2. [Manual model development](#manual_model_development)
# ### 3. [AutoML development](#automl_development)
# ### 4. [Model Training and Testing](#model_training_and_testing)

# ---
# ---

# # Dataset formatting and feature extraction<a id='dataset_formatting_and_extraction'></a>

# ## a. Setup the links to the data file to be used for model training and testing

# In[1]:


# System library for path management
from os import path

# Set the patha for the training and test datafiles
user_home_dir = str(path.expanduser('~'))
print('Home directory for the current user : ', user_home_dir)


# In[2]:


# Add path to the sample data file for training and testing models
sample_file_path = path.join(user_home_dir,
                                  'Desktop\MLH-2018\\amex-nyse-nasdaq-stock-histories\subset_data\AAL.csv')
print('Path to the data file currently being used : ', sample_file_path)


# ## b. Read the data file into a dataframe to be pre-processed

# In[3]:


# Import the system built-in modules needed for feature extraction
import os
import time
from datetime import datetime


# In[4]:


# Import the essential data processing libraries
import pandas as pd
import numpy as np


# In[5]:


# Import visualization libraries for plotting and visualizing the dataset 
# vectors
import matplotlib.pyplot as plt

# In[7]:


# Read in the dataset from the csv file, and convert to a Pandas dataframe
sample_dataframe = pd.read_csv(sample_file_path, engine='python', encoding='utf-8 sig')
print(sample_dataframe)


# In[8]:


# Convert the date format in the dataframe into POSIX Timestamps

default_timestamps = sample_dataframe['date'].values
show_values = 5

# Initialize the list for storing POSIX timestamps
posix_timestamps = []

# Transform the datetime into POSIX datetime
for i in range(default_timestamps.shape[0]):
    
    # Collect the logged time value
    timestamp_logged = default_timestamps[i]    
    
    # Convert the logged default timestamp to POSIX and add to the list
    posix_timestamps.append(datetime.strptime(timestamp_logged, '%Y-%m-%d'))
    posix_timestamps[i] = time.mktime(posix_timestamps[i].timetuple())

# Add the list to the dataframe
sample_dataframe['Timestamp'] = posix_timestamps

# Set the POSIX timestamp column to be the index of the dataframe
#sample_dataframe.set_index('Timestamp', inplace=True)

# Sort the POSIX timestamp values in the dataframe
sample_dataframe.sort_values(by=['Timestamp'], inplace=True)

# Give a preview of the re-index dataframe
print('Showing the first %d values from the dataframe.' %(show_values))
sample_dataframe.head(show_values)


# ## Plot the values in the given dataset

# In[9]:


# Extract the values to plot from the dataframe
timestamps = sample_dataframe['Timestamp'].values

df_cols = list(sample_dataframe.columns.values)
df_cols.remove('date')
df_cols.remove('Timestamp')
df_cols.remove('volume')

data_values = []
for col in df_cols:
    values = list(sample_dataframe[col].values)
    data_values.append(values)
 
print(len(data_values))


# In[10]:


# Import the plotting library
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('original_dataset.png', dpi=100)
fig.set_size_inches(18.5, 10.5, forward=True)

for i in range(len(data_values)):
    plt.plot(timestamps, data_values[i])

plt.legend(df_cols)
plt.show()
#plt.plot(timestam)


# In[11]:


fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('original_dataset_volumne.png', dpi=100)
fig.set_size_inches(18.5, 10.5, forward=True)

plt.plot(timestamps, sample_dataframe['volume'])
plt.legend('Volume')
plt.show()


# ## Now that the dataframe are constructed, we can start the feature extraction and normalization

# In[12]:


# Initialize the list to hold the normalized data valuess
scaled_features = []

# Approach 1 : Using the data from previous 7 days to predict the next day
from sklearn.preprocessing import MinMaxScaler

# Normalize and scale each one of the input features and construct the 
# scaler array for doing inverse scaling later
sample_df_cols = list(sample_dataframe.columns.values)
sample_df_cols.remove('date')

scaler_array = []
scaled_features_dict = {}

for col in sample_df_cols:
    
    # Initialize the scaler for the given feature
    # Range : -1...1
    scaler = MinMaxScaler(feature_range=(-1,1), copy=True)
    
    # Extract the features and fit the scaler
    feature_list = sample_dataframe[col].values.reshape(-1,1)
    scaler.fit(feature_list)
    
    # Add th scaler to the dictionary
    scaler_array.append(scaler)
    
    # Transform the feature dataset
    scaled_feature_list = scaler.transform(feature_list)
    scaled_features.append(scaled_feature_list)
    scaled_features_dict[col] = scaled_feature_list
    
print(scaled_features)


# In[13]:


# Construct the Numpy array to hold the data values
scaled_features = np.array(scaled_features)
scaled_features = np.transpose(scaled_features).reshape(scaled_features.shape[0], scaled_features.shape[1])
scaled_features = np.transpose(scaled_features)

print(scaled_features.shape)
# Determine the dimenstions on the input dataset
print('Dimensions of the input dataset : ', scaled_features.shape)


# ## Split the dataset into training and testing 

# In[14]:


def make_dataset_batchable(data_array_in, desired_ratio, batch_size):
      
      # Length of the input dataset and number of batches available
      data_length = data_array_in.shape[0]
      num_batches = int(data_length/batch_size)
      
      # Length of the usable dataset with the given batches
      data_use_len = num_batches * batch_size
      
      if data_use_len < data_length:
          # Format and remove the extra datapoints from the datasets
          actual_data_in = np.delete(data_array_in, np.s_[data_use_len::], axis=0)
      else:
          actual_data_in = data_array_in
      
      # Size of training and testing sets initially available
      train_length = int(desired_ratio * data_use_len)
      test_length = int((1 - desired_ratio) * data_use_len)
      
      # Number of testing and training initially batches available
      num_train_batches = int(train_length / batch_size)
      num_test_batches = int(test_length / batch_size)
      
      # NUmber of data points not used for training and testing
      leftover = data_use_len - (num_train_batches*batch_size) - (
              num_test_batches*batch_size)
      
      # Number of batches available after initial splitting
      leftover_batches = leftover / batch_size
      
      # Calculate the best ratio to use and increase the training size
      actual_ratio = float((batch_size*num_train_batches)+(batch_size*
                      leftover_batches)) / float(data_use_len)
      
      return actual_ratio, actual_data_in


# In[15]:


# Make the dataset batchable for the training and testing

batch_size = 7
test_train_ratio = 0.80

ratio_to_use, scaled_features = make_dataset_batchable(scaled_features,
                                                      test_train_ratio,
                                                      batch_size)


# In[16]:


# Determine the dimenstions on the input dataset
print('Dimensions of the input dataset : ', scaled_features.shape)


# ## Now that the input dataset is guaranteed to be batchable, create the training and testing batches

# In[17]:


def split_dataset(data_set_in, train_test_ratio):
      
     # Determine the length of the training and testing arrays
     train_size = int(len(data_set_in) * train_test_ratio)
    
     # Split the dataset
     train, test = data_set_in[0:train_size][:], data_set_in[train_size:len(data_set_in)][:]
     
     return train, test


# In[18]:


train_dataset, test_dataset = split_dataset(scaled_features, ratio_to_use)


# In[19]:


# Display the training dataset to be used for training the network
print('Displaying the training dataset with size : ', train_dataset.shape)
print(train_dataset)


# In[20]:


# Display the testing dataset to be used for testing the network
print('Displaying the testing dataset with size : ', test_dataset.shape)
print(test_dataset)


# ## Now that the features have been scaled, construct the supervised dataset for generating predictions

# ## Reshape the input dataset

# In[21]:


# Construct the supervised dataset for training the models

# Define the dimenstions of the input dataset to be used for training the
# model

num_past_timestamps = 7
num_future_predictions = 1
num_features = len(sample_df_cols)
num_samples = len(scaled_features[0])

# Method for converting the input dataset to a supervised training dataset
from pandas import DataFrame
from pandas import concat
 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[22]:


training_input_data = series_to_supervised(data=train_dataset,
                                       n_in=num_past_timestamps,
                                      n_out=num_future_predictions)


# In[23]:


test_input_data = series_to_supervised(data=test_dataset,
                                       n_in=num_past_timestamps,
                                      n_out=num_future_predictions)


# In[24]:


print(training_input_data)


# In[25]:


print(test_input_data)


# In[26]:


# Reshape the supervised dataset into the 3-D format for the LSTM
# Input format : [samples, timestamps, features]
training_input_data = np.array(training_input_data)
print(training_input_data)
training_input_data = training_input_data.reshape(training_input_data.shape[0],
                                         num_past_timestamps+1,
                                         num_features)


# In[27]:


print('Displaying the input training dataset ready for input to RNNs : ')
print(training_input_data)


# In[28]:


# Display the dimensions of the input training and testing data
print('Shape of training data :', training_input_data.shape[0])


# In[29]:


# Reshape the supervised dataset into the 3-D format for the LSTM
# Input format : [samples, timestamps, features]
test_input_data = np.array(test_input_data)
print(test_input_data)
test_input_data = test_input_data.reshape(test_input_data.shape[0],
                                         num_past_timestamps+1,
                                         num_features)


# In[30]:


print('Displaying the input testing dataset ready for input to RNNs : ')
print(test_input_data)


# ## Now that the input dataset is ready, construct the output dataset for training and testing

# In[31]:


# Create the output dataset for the training and testing the model
def create_output(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(1, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[32]:


# Create the train and test output dataset

train_output = create_output(data=train_dataset,
                            n_in=0,
                            n_out=2,
                            dropnan=True)


train_output.drop(index=[0,1,2,3,4,5], inplace=True)
print(train_output)

train_output


# In[33]:


test_output = create_output(data=test_dataset,
                            n_in=0,
                            n_out=2,
                            dropnan=True)


test_output.drop(index=[0,1,2,3,4,5], inplace=True)
print(test_output)


# In[34]:


# Extract the data from the dataframes
train_output_np = train_output.values
test_output_np = test_output.values

print(train_output_np)
print(test_output_np)


# In[35]:


print(train_output_np.shape[0])


# # Manual model development<a id='manual_model_development'></a>

# ## Model 1 : LSTM Neural Network
# 
# ### Goal - Initialize and implement the manual LSTM model and train using the given dataset

# In[36]:


# Layer and Model Initializers from Keras
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional, GRU, Dropout

# Visualizers for the model
from keras.utils import plot_model

# Optimizizers for training and network performance
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop, SGD


# In[ ]:


"""# fit an LSTM network to training data
def fit_lstm(train_in, train_out, batch_size, nb_epoch, neurons):
    X, y = train_in, train_out
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(train_out.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model"""


# In[ ]:


"""# Initialize the neural network model
opt = SGD(lr=0.05, nesterov=True)
batch_size = 4
past_timesteps = 1

# Define the network model and layers
input_1 = Input(batch_shape=(batch_size, past_timesteps))
dense_2 = LSTM(units=5, activation='relu')(input_1)
dense_3 = Dense(units=10, activation='relu')(dense_2)
dense_6 = Dense(units=10, activation='sigmoid')(dense_3)
dense_7 = Dense(units=5, activation='relu')(dense_6)
output_1 = Dense(units=1)(dense_7)
        
# Generate and compile the model
predictor = Model(inputs=input_1, outputs=output_1)
predictor.compile(optimizer= opt, loss= 'mae', metrics=['mape', 'mse', 'mae'])
predictor.summary()"""


# In[ ]:


opt = Adam(lr=0.05)
batch_size = 1
past_timestamps = 8
num_features = len(sample_df_cols)

# Define the network model and layers
input_1 = Input(batch_shape=(batch_size, past_timestamps, num_features))
lstm_1 = LSTM(units=52, stateful=True, return_sequences=True)(input_1)
#dropout_1 = Dropout(0.2, noise_shape=None, seed=None)(lstm_1)
lstm_2 = LSTM(units=52, stateful=True)(lstm_1)
dense_1 = Dense(units=30, activation='sigmoid')(lstm_2)
dense_2 = Dense(units=5)(dense_1)
output_1 = Dense(units=num_features)(dense_2)

# Generate and compile the model
predictor = Model(inputs=input_1, outputs=output_1)
predictor.compile(optimizer=opt, loss='mse', metrics=['mape',
                                                     'mse',
                                                     'mae'])
predictor.summary()


# In[ ]:


num_epochs = 200
predictor.fit(x=training_input_data, y=train_output_np, 
              batch_size=batch_size, epochs=num_epochs, verbose=2,
             shuffle=False)


# In[ ]:


test_predictions = predictor.predict(x=test_input_data,
                                    batch_size=batch_size, verbose=2)


# In[ ]:


print(test_predictions)


# In[ ]:


print(test_predictions.shape)


# In[ ]:


# Need to create a transpose of the 2-D array for inverse scaling
test_output = np.transpose(test_predictions)

for i in range(test_output.shape[0]):
    test_output[i].reshape(-1,1)
    
print(test_output)
print(test_output.shape)


# In[ ]:


transformed_values = []

for i in range(test_output.shape[0]):
    values = np.array(list(test_output[i]))
    values = values.reshape(-1,1)
    transformed_values.append(scaler_array[i].inverse_transform(values))
    
print(transformed_values)


# In[ ]:


# Generate np array for the predictions
transformed_values_np = np.array(transformed_values)
print(transformed_values_np.shape)


# In[ ]:


# Plot the predictions generated by the neural network
predictions_df = pd.DataFrame(
    transformed_values_np.reshape(transformed_values_np.shape[0],
                              transformed_values_np.shape[1]))
print(predictions_df)


# In[ ]:


transformed_values_np = transformed_values_np.reshape(transformed_values_np.shape[0],
                              transformed_values_np.shape[1])
print(transformed_values_np)
print(transformed_values_np.shape)

# Save the predictions on the FS as a Dataframe
save_df = pd.DataFrame(data=transformed_values_np)

predictions_save_path = path.join(user_home_dir,
                                  'Desktop\MLH-2018\\amex-nyse-nasdaq-stock-histories\subset_data\AAL-P.csv')
                                

save_df.to_csv(predictions_save_path, sep=',')

# In[ ]:


transformed_values_array = list(transformed_values_np)
print(len(transformed_values_array))
print(len(transformed_values_array[0]))
#for i in range(transformed_values_np.shape[0]):
    #transformed_values_array.append(transformed_values_np[i])
print(transformed_values_array)


# In[ ]:


volume_predictions = transformed_values_array[0]
print(len(volume_predictions))


# In[ ]:


volume_actual = sample_dataframe['volume'].values[train_output_np.shape[0]+20:]
timestamps_predictions = timestamps[train_output_np.shape[0]+20:]

print(len(volume_actual))
print(len(timestamps_predictions))

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('volume_predictioss.png', dpi=100)
fig.set_size_inches(18.5, 10.5, forward=True)

plt.plot(timestamps_predictions, volume_predictions)
plt.plot(timestamps_predictions, volume_actual)

plt.legend(['Predictions', 'Actual'])
plt.show()
#plt.plot(timestam)


# In[ ]:


volume_predictions = transformed_values_array[1]
print(len(volume_predictions))

volume_actual = sample_dataframe['open'].values[train_output_np.shape[0]+20-6:]
timestamps_predictions = timestamps[train_output_np.shape[0]+20-6:]

print(len(volume_actual))
print(len(timestamps_predictions))

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('open_predictions.png', dpi=10)
fig.set_size_inches(18.5, 10.5, forward=True)

plt.plot(timestamps_predictions, volume_predictions)
plt.plot(timestamps_predictions, volume_actual)

plt.legend(['Predictions', 'Actual'])
plt.show()
#plt.plot(timestam)


# In[ ]:


volume_predictions = transformed_values_array[2]
print(len(volume_predictions))

volume_actual = sample_dataframe['close'].values[train_output_np.shape[0]+20:]
timestamps_predictions = timestamps[train_output_np.shape[0]+20:]

print(len(volume_actual))
print(len(timestamps_predictions))

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('close_predictions.png', dpi=10)
fig.set_size_inches(18.5, 10.5, forward=True)

plt.plot(timestamps_predictions, volume_predictions)
plt.plot(timestamps_predictions, volume_actual)

plt.legend(['Predictions', 'Actual'])
plt.show()


# # Check how the model did on the training data

# In[ ]:


batch_size = 1
train_predictions = predictor.predict(x=training_input_data,
                                    batch_size=batch_size, verbose=2)


# In[ ]:


# Need to create a transpose of the 2-D array for inverse scaling
train_output = np.transpose(train_predictions)

for i in range(train_output.shape[0]):
    train_output[i].reshape(-1,1)
    
print(train_output)
print(train_output.shape)


# In[ ]:


transformed_values_2 = []

for i in range(train_output.shape[0]):
    values = np.array(list(train_output[i]))
    values = values.reshape(-1,1)
    transformed_values_2.append(scaler_array[i].inverse_transform(values))
    
print(transformed_values_2)


# In[ ]:


# Generate np array for the predictions
transformed_values_np_2 = np.array(transformed_values_2)
print(transformed_values_np_2.shape)


# In[ ]:


transformed_values_array_2 = list(transformed_values_np_2)
print(len(transformed_values_array_2))
print(len(transformed_values_array_2[0]))
#for i in range(transformed_values_np.shape[0]):
    #transformed_values_array.append(transformed_values_np[i])
print(transformed_values_array_2)


# In[ ]:


volume_predictions = transformed_values_array_2[0]



# In[ ]:


volume_actual = sample_dataframe['volume'].values[:len(volume_predictions)]
timestamps_predictions = timestamps[:len(volume_predictions)]

print(len(volume_actual))
print(len(timestamps_predictions))

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('volume_predictions.png', dpi=10)
fig.set_size_inches(18.5, 10.5, forward=True)

plt.plot(timestamps_predictions, volume_predictions)
plt.plot(timestamps_predictions, volume_actual)

plt.legend(['Predictions', 'Actual'])
plt.show()

#--------------------------------------------------------------------------------------------------


