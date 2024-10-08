import numpy as np
import random

def TTS(X, y, test_size=0.2, shuffle=True):
    if shuffle:
        combined = list(zip(X, y))
        random.shuffle(combined)
        X, y = zip(*combined)
    
    test_size = int(len(X) * test_size)
    
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)