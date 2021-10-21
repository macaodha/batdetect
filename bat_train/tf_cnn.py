import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import confusion_matrix

def cnn_all(train_ds, test_ds, params, input_shape, size = 'small', save_dir=''):

    
    if size == 'small':
        model = get_cnn_small(input_shape)
    if size == 'big':
        model = get_cnn_big(input_shape)

    print(model.summary())

    model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss      = tf.keras.losses.SparseCategoricalCrossentropy(),#from_logits=True),
    metrics   = 'accuracy',#tf.keras.metrics.SparseCategoricalAccuracy(),
    )
    
    history = model.fit(train_ds, 
    validation_data = test_ds,  
    epochs          = params.num_epochs,
    callbacks       = tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
    )

    metrics = history.history

    fig = go.Figure()

    fig.add_trace(go.Scatter(x = history.epoch, y = metrics['loss'], 
                            mode = 'lines', name = 'Training',
                            line = dict(color='dodgerblue')))
    fig.add_trace(go.Scatter(x = history.epoch, y = metrics['val_loss'], 
                            mode = 'lines', name = 'Validation',
                            line = dict(color='orange')))
    fig.update_layout(title       = 'Loss Curve',
                      xaxis_title = 'Epoch',
                      yaxis_title = 'Loss')

    print('Saving CNN Loss curve')
    fig.write_image(save_dir + '_'+size+'_cnn_loss.pdf')
    fig.write_html(save_dir  + '_'+size+'_cnn_loss.html')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = history.epoch, y = metrics['accuracy'], 
                            mode = 'lines', name = 'Training',
                            line = dict(color='dodgerblue')))
    fig.add_trace(go.Scatter(x = history.epoch, y = metrics['val_accuracy'], 
                            mode = 'lines', name = 'Validation',
                            line = dict(color='orange')))
    fig.update_layout(title       = 'Accuracy Curve',
                      xaxis_title = 'Epoch',
                      yaxis_title = 'Accuracy')

    print('Saving CNN Accuracy curve')
    fig.write_image(save_dir + '_'+size+'_cnn_acc.pdf')
    fig.write_html(save_dir  + '_'+size+'_cnn_acc.html')

    return model

def get_cnn_small(input_shape):
    mod = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(2, activation = 'softmax'),
    ])
    return mod

def get_cnn_big(input_shape):
    mod = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding = 'same'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu', padding = 'same'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu', padding = 'same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(2, activation = 'softmax'),
    ])
    return mod

def prec_recall_curves(y_true, y_test_probs):
    # Containers for true positive / false positive rates
    precision_scores = []
    recall_scores    = []

    # Define probability thresholds to use, between 0 and 1
    probability_thresholds = np.linspace(0, 1, num=100)

    # Find true positive / false positive rate for each threshold
    for p in probability_thresholds:
        y_test_preds =1*(y_test_probs >= p)

        precision, recall = prec_rec_vals(confusion_matrix(y_true, y_test_preds))
        precision_scores.append(precision)
        recall_scores.append(recall)
    return precision_scores, recall_scores

def prec_rec_vals(cm): 
    tn, fp, fn, tp    = cm.ravel()
    precision, recall = tp/(tp+fp), tp/(tp+fn)
    return(precision, recall)