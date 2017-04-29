#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking


def custom_model(data_shape, layers=1, units1=128, units2=128, units3=128, units4=128, units5=128, optim='rmsprop'):
    units = {
        1: units1,
        2: units2,
        3: units3,
        4: units4,
        5: units5
    }

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=data_shape))
    return_sequences = True
    for i in range(layers):
        if i+1 >= layers:
            return_sequences = False
        if i == 0:
            model.add(LSTM(units[i+1], input_shape=data_shape,
                return_sequences=return_sequences))
        else:
            model.add(LSTM(units[i+1], return_sequences=return_sequences))

    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])

    return model


def light_model(data_shape, units1=81, optim='adam'):
    model = Sequential()
    model.add(LSTM(units1, input_shape=data_shape))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    return model


def medium_model(data_shape, units1=128, units2=64, units3=32, optim='rmsprop'):
    model = Sequential()
    model.add(LSTM(units1, return_sequences=True,
                   input_shape=data_shape))
    model.add(LSTM(units2, return_sequences=True))
    model.add(LSTM(units3))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    return model


def heavy_model(data_shape, units1=512, units2=256, units3=128, units4=64, units5=32, optim='rmsprop'):
    model = Sequential()
    model.add(LSTM(units1, return_sequences=True,
                   input_shape=data_shape))
    model.add(LSTM(units2, return_sequences=True))
    model.add(LSTM(units3, return_sequences=True))
    model.add(LSTM(units4, return_sequences=True))
    model.add(LSTM(units5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    return model

