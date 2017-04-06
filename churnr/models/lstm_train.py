#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
import os
import logging
import argparse
import joblib
import json

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

from churnr.models.lstm_models import light_model
from churnr.utils import plot_roc_auc, yes_or_no


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('churnr.lstm')


def main(exppath, experiment, dsname, modelname):
    logger.info('Initializing LSTM training...')

    with open(exppath) as fi:
        expconf = json.load(fi)[experiment]

    # load experiment configuration
    keys = ['procpath']
    conf = {}
    for key in keys:
        conf[key] = expconf['datasets'][dsname][key] if key in expconf['datasets'][dsname] else expconf['datasets']['global'][key]
    keys = ['modelpath']
    for key in keys:
        conf[key] = expconf['models'][modelname][key] if key in expconf['models'][modelname] else expconf['models']['global'][key]

    procpath = conf['procpath']
    modelpath = conf['modelpath']
    Xpath_tr = os.path.join(procpath, experiment, dsname, 'features_seq_train.gz')
    ypath_tr = os.path.join(procpath, experiment, dsname, 'labels_train.gz')
    Xpath_te = os.path.join(procpath, experiment, dsname, 'features_seq_test.gz')
    ypath_te = os.path.join(procpath, experiment, dsname, 'labels_test.gz')

    logger.info('Loading features from [{}] and targets from [{}]'.format(Xpath_tr, ypath_tr))

    X = joblib.load(Xpath_tr)
    y = to_categorical(joblib.load(ypath_tr))
    X_te = joblib.load(Xpath_te)
    y_te = to_categorical(joblib.load(ypath_te))
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # get the model instance
    logger.info('Compiling LSTM model...')
    model = light_model(data_shape=(X.shape[1], X.shape[2]))

    modeldir = os.path.join(modelpath, experiment, dsname, modelname)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    try:
        # set model callbacks
        chkp_path = os.path.join(modeldir, 'model_weights.hdf5')
        chkp = ModelCheckpoint(chkp_path, monitor='val_acc', save_best_only=True, period=1, verbose=1)
        reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=3, min_lr=0.001, verbose=1, cooldown=10, epsilon=0.1)
        tensorboard = TensorBoard(log_dir=os.path.join(modeldir, 'logs'), write_images=True)
        earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=20, verbose=1)
        callbacks = [chkp, reducelr, tensorboard, earlystop]

        # train the model for each train / val folds
        logger.info('Training model...')

        model.fit(X, y,
                    batch_size=2048, epochs=150,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks)

        # reload the best model checkpoint
        logger.info('Reloading checkpoint from best model found...')
        model = load_model(chkp_path)

        # print the model test metrics
        logger.info('Evaluating model on the test set...')
        metrics_values = model.evaluate(X_te, y_te)
        metrics_names = model.metrics_names if type(model.metrics_names) == list else [model.metrics_names]
        metrics = {}
        logger.info('** Test metrics **')
        for metric, metric_name in zip(metrics_values, metrics_names):
            logger.info('-- {0}: {1:.3f}'.format(metric_name, metric))
            metrics[metric_name] = metric

        # calculate roc and auc and plot it
        y_pred = model.predict(X_te)
        roc_auc = plot_roc_auc(y_te[:,1], y_pred[:,1], modeldir)
        logger.info('-- auc: {0:.3f}'.format(roc_auc))
        metrics['auc'] = roc_auc

        # serialize y_true and y_pred for later roc visualization
        y_trpred = np.empty(shape=(y_te.shape[0], 2))
        y_trpred[:,0] = y_te[:,1]
        y_trpred[:,1] = y_pred[:,1]
        joblib.dump(y_trpred, os.path.join(modeldir, 'y_test_true_pred.gz'))

        # save the metrics and config
        with open(os.path.join(modeldir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        with open(os.path.join(modeldir, 'config.json'), 'w') as f:
            json.dump(model.get_config(), f)

    except Exception as e:
        logger.exception(e)
        ans = yes_or_no('Delete folder at {}?'.format(modeldir))
        if ans:
            import shutil
            shutil.rmtree(modeldir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM trainer')
    parser.add_argument('--exppath', default='../../experiments.json', help='Path to the experiments json file')
    parser.add_argument('--experiment', default='temporal_static', help='Name of the experiment being performed')
    parser.add_argument('--dsname', default='session_6030d', help='Name of the dataset used for training')
    parser.add_argument('--modelname', default='lstm', help='Name of the model being trained')

    args = parser.parse_args()

    main(exppath=args.exppath, experiment=args.experiment, dsname=args.dsname, modelname=args.modelname)

