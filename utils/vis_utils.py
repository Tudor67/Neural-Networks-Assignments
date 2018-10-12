import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix


def plot_train_val_loss_acc(train_loss, val_loss, train_acc, val_acc, title='sgd'):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='train')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss [{}]'.format(title))

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='train')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy [{}]'.format(title))


def plot_confusion_matrix(y_pred, y, classes,
                          normalize=False,
                          title='Confusion matrix',
                          new_figure=False,
                          cmap=plt.cm.Blues):
    '''
    From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#confusion-matrix
    '''
    cm = confusion_matrix(y, y_pred).T
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if new_figure:
        plt.figure(figsize=(8, 6))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Predicted label')
    plt.xlabel('True label')
