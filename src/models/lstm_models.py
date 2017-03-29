#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import LSTM, Dense


def light_model(data_shape, units1=64, optim='rmsprop'):
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

