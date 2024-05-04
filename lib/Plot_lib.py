import numpy as np
from matplotlib import pyplot as plt


def plot_dataset(data, x, x_label, y_1, y_label_1, y_2, y_label_2, y_label, title):
    plt.figure(figsize=(14, 5), dpi=100)
    plt.plot(data[x], data[y_1], label=y_label_1)
    plt.plot(data[x], y_2, label=y_label_2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_dataset(data, x, x_label, y, y_label_1, y_label, title):
    plt.figure(figsize=(14, 5), dpi=100)
    plt.plot(data[x], data[y], label=y_label_1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_single_col(data, length, x_label, y_label_1, y_label, title):
    plt.figure(figsize=(14, 5), dpi=100)
    plt.plot(np.arange(0, length), data, label=y_label_1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_technical_indicators(dataset, time_steps):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0 - time_steps

    dataset = dataset.iloc[-time_steps:, :]
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)

    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['close'], label='Closing Price', color='b')
    plt.plot(dataset['SMA20'], label='SMA 100m', color='g', linestyle='--')
    plt.plot(dataset['SMA50'], label='SMA 250m', color='r', linestyle='--')
    plt.plot(dataset['Upperband12'], label='Upper Band 1h', color='c')
    plt.plot(dataset['Lowerband12'], label='Lower Band 1h', color='c')
    plt.fill_between(x_, dataset['Lowerband12'], dataset['Upperband12'], alpha=0.15)
    plt.title('Technical indicators for BTC - last {} Time intervals.'.format(time_steps))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD12')
    plt.plot(dataset['MACD12'], label='MACD 1h', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['MOM10'], label='Momentum 10', color='b', linestyle='-')

    plt.legend()
    plt.show()
