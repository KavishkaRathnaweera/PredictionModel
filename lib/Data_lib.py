import numpy as np
import talib as tb
import pandas as pd


def NaNCheck(data):
    return data.isnull().sum()[data.isnull().sum() > 0]


## Create Indicators
# Simple Moving Average :
# *   SMA3 - 15m
# *   SMA12 - 1h
# *   SMA20 - 100m
# *   SMA48 - 4h
# *   SMA50 - 250m
# *   SMA288 - 1d
#
# Exponential Moving Average
# *   EMA12 - 1h
# *   EMA20 - 100m
# *   EMA26 - 130m
#
# Moving Average Convergence/Divergence
# *   MACD12 - 1h
# *   MACD24 - 2h
#
# Relative Strength Index
# *   RSI12 - 1h
# *   RSI24 - 2h
# *   RSI36 - 3h
#
# Average True Range
# *   ATR12 - 1h
# *   ATR24 - 2h
# *   ATR36 - 3h
#
# Bollinger Bands
# *   UpperBand12,LowerBand12  - 1h
# *   UpperBand24,LowerBand24 - 2h
# *   UpperBand36,LowerBand36 - 3h
#
# MOM
# *   MoM10
#
# OBV
# *   OBV
#
# https://mrjbq7.github.io/ta-lib/
#
# https://www.forex.in.rs/moving-average-for-5-min-chart/#:~:text=The%20best%20moving%20averages%20for,20%20MA%20and%2050%20MA.
def CreateInc(dataset, close, high, low, volume):
    new_df = dataset.copy()

    # Moving Average
    new_df["SMA3"] = tb.SMA(dataset[close], timeperiod=3)
    new_df["SMA12"] = tb.SMA(dataset[close], timeperiod=12)
    new_df["SMA20"] = tb.SMA(dataset[close], timeperiod=20)
    new_df["SMA48"] = tb.SMA(dataset[close], timeperiod=48)
    new_df["SMA50"] = tb.SMA(dataset[close], timeperiod=50)
    new_df["SMA288"] = tb.SMA(dataset[close], timeperiod=288)
    m1 = max(new_df[new_df['SMA288'].isnull()].index)

    # Exponential Moving Average
    new_df["EMA12"] = tb.EMA(dataset[close], timeperiod=12)
    new_df["EMA20"] = tb.EMA(dataset[close], timeperiod=20)
    new_df["EMA26"] = tb.EMA(dataset[close], timeperiod=26)
    m2 = max(new_df[new_df['EMA26'].isnull()].index)

    # Moving Avarage Convergence Divergernce
    MACD12, macdsignal12, macdhist12 = tb.MACD(dataset[close], fastperiod=18, slowperiod=6, signalperiod=12)
    MACD24, macdsignal24, macdhist24 = tb.MACD(dataset[close], fastperiod=36, slowperiod=12, signalperiod=24)
    new_df["MACD12"] = MACD12
    new_df["MACD24"] = MACD24
    m3 = max(new_df[new_df['MACD24'].isnull()].index)

    # Relative Strength Index
    new_df["RSI12"] = tb.RSI(dataset[close], timeperiod=12)
    new_df["RSI24"] = tb.RSI(dataset[close], timeperiod=24)
    new_df["RSI36"] = tb.RSI(dataset[close], timeperiod=36)
    m4 = max(new_df[new_df['RSI36'].isnull()].index)

    # Average True Range
    new_df["ATR12"] = tb.ATR(dataset[high], dataset[low], dataset[close], timeperiod=12)
    new_df["ATR24"] = tb.ATR(dataset[high], dataset[low], dataset[close], timeperiod=24)
    new_df["ATR36"] = tb.ATR(dataset[high], dataset[low], dataset[close], timeperiod=36)
    m5 = max(new_df[new_df['ATR36'].isnull()].index)

    # Bolinger Bands
    upperband12, middleband12, lowerband12 = tb.BBANDS(dataset[close], timeperiod=12, nbdevup=2, nbdevdn=2, matype=0)
    upperband24, middleband24, lowerband24 = tb.BBANDS(dataset[close], timeperiod=24, nbdevup=2, nbdevdn=2, matype=0)
    upperband36, middleband36, lowerband36 = tb.BBANDS(dataset[close], timeperiod=36, nbdevup=2, nbdevdn=2, matype=0)
    new_df["Upperband12"] = upperband12
    new_df["Lowerband12"] = lowerband12
    new_df["Upperband24"] = upperband24
    new_df["Lowerband24"] = lowerband24
    new_df["Upperband36"] = upperband36
    new_df["Lowerband36"] = lowerband36
    m6 = max(new_df[new_df['Upperband36'].isnull()].index)

    # Momentum
    new_df["MOM10"] = tb.MOM(dataset[close], timeperiod=10)
    m7 = max(new_df[new_df['MOM10'].isnull()].index)

    # Volume Indicators
    new_df["OBV"] = tb.OBV(dataset[close], dataset[volume])
    # m8 = max(new_df[new_df["OBV"].isnull()].index)

    # Remove NaN values
    m = max(m1, m2, m3, m4, m5, m6, m7)
    new_df = new_df.loc[m + 1:]
    new_df = new_df.reset_index()
    new_df = new_df.drop(columns=["index"])

    return new_df


def multi_freq_pattern(dataset):
    close_fft = np.fft.fft(np.asarray(dataset['Close'].tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    output_df_fft = fft_df.copy()
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9, 27, 81, 100]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        output_df_fft[f'FT_{num_}components'] = np.fft.ifft(fft_list_m10).astype('float')
    return output_df_fft
