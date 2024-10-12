import numpy as np

class Perceptron:
    def __init__(self, lr_w, lr_b, epochs):
        self.learning_rate_w = lr_w
        self.learning_rate_b = lr_b
        self.epochs = epochs
        self.weights_history = []
        self.biases_history = []
        self.losses = []

    def fit(self, X_train, y_train):
        # تعداد ویژگی‌ها را از X_train بگیرید
        self.w = np.random.rand(X_train.shape[1], 1)  
        self.b = np.random.rand()

        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                x = X_train[i].reshape(1, -1)  # تغییر شکل به (1, n_features)
                y = y_train[i].reshape(-1, 1)   # تغییر شکل به (1, 1)

                y_pred = x @ self.w + self.b
                error = y - y_pred

                # به‌روزرسانی وزن‌ها و بایاس
                self.w += self.learning_rate_w * error * x.T
                self.b += self.learning_rate_b * error.flatten()  # تبدیل به یک بعدی

            self.weights_history.append(self.w.copy())
            self.biases_history.append(self.b)
            loss = self.evaluate(X_train, y_train, "mae")
            self.losses.append(loss)

            print(f"Epoch {epoch + 1} done. Loss: {loss}")

    def predict(self, X_test):
        return (X_test @ self.w + self.b).flatten()  # تبدیل به یک بعدی

    def evaluate(self, X_test, y_test, metric: str):
        y_pred = X_test @ self.w + self.b
        error = y_test.reshape(-1, 1) - y_pred

        if metric == "mae":
            loss = np.mean(np.abs(error))
        elif metric == "mse":
            loss = np.mean(error ** 2)
        elif metric == "rmse":
            loss = np.sqrt(np.mean(error ** 2))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return loss
