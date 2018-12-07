# -*- coding: utf-8 -*-
from tkinter import *
import pandas as pd
from datetime import datetime
from os import path
import time
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

stock_name = ""
date_start = ""
date_end = ""

def clicked():
    stock_name = txt.get()
    date_start = txt2.get()
    date_end = txt3.get()

    # Convert the logged default timestamp to POSIX and add to the list
    posix_timestamp_1 = datetime.strptime(date_start, '%Y-%m-%d')
    posix_timestamp_2 = datetime.strptime(date_end, '%Y-%m-%d')

    posix_timestamp_1 = time.mktime(posix_timestamp_1.timetuple())
    posix_timestamp_2 = time.mktime(posix_timestamp_2.timetuple())

    print(posix_timestamp_1)
    print(posix_timestamp_2)

    user_home_dir = str(path.expanduser('~'))
    dataframe_path = path.join(user_home_dir,
                                  'Desktop\MLH-2018\\amex-nyse-nasdaq-stock-histories\subset_data\AAL-P.csv')
    orig_dataframe_path = path.join(user_home_dir,
                                  'Desktop\MLH-2018\\amex-nyse-nasdaq-stock-histories\subset_data\AAL.csv')

    df = pd.read_csv(dataframe_path)
    df_orig = pd.read_csv(orig_dataframe_path)

    # Convert the date format in the dataframe into POSIX Timestamps

    default_timestamps = df_orig['date'].values

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
    df_orig['Timestamp'] = posix_timestamps
    df_orig.sort_values(by=['Timestamp'], inplace=True)
    
    time_series = df_orig['Timestamp'].values
    start_index = pd.Index(time_series).get_loc(posix_timestamp_1)
    end_index = pd.Index(time_series).get_loc(posix_timestamp_2)

    volume_actual = df_orig['volume'].values[start_index:end_index]
    timestamps_predictions = posix_timestamps[start_index:end_index]

    print(len(volume_actual))
    print(len(timestamps_predictions))

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('volume_predictions.png', dpi=20)
    fig.set_size_inches(18.5, 10.5, forward=True)

    #plt.plot(timestamps_predictions, volume_predictions)
    plt.plot(timestamps_predictions, volume_actual)

    plt.legend(['Predictions', 'Actual'])
    plt.show()

    print(start_index)
    print(end_index)

if __name__ == '__main__':

    window = Tk()
    window.title("Stock Price Predictor")

    window.geometry('600x200')

    lbl = Label(window, text="Enter the Stock ID here (e.g. AAL) : ")
    lbl.grid(column=350, row=0)

    txt = Entry(window, width=10)
    txt.grid(column=350, row=200)

    lbl2 = Label(window, text="Enter the start date here : ")
    lbl2.grid(column=300, row=400)

    txt2 = Entry(window, width=10)
    txt2.grid(column=300, row=600)

    lbl3 = Label(window, text="Enter the end date here : ")
    lbl3.grid(column=600, row=400)

    txt3 = Entry(window, width=10)
    txt3.grid(column=600, row=600)

    btn = Button(window, text="Enter", command=clicked)
    btn.grid(column=350, row=800)
    #button = Button(window, text='Enter', command=clicked)

    window.mainloop()

