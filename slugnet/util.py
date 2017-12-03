def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))
