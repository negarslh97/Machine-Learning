import numpy as np
import cv2
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        Y_pred = []
        for x in X:
            distances = self.euclidean_distance(x, self.X_train)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.Y_train[k_indices]
            Y_pred.append(np.bincount(k_nearest_labels).argmax())
        return np.array(Y_pred)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.sum(Y_pred == Y) / len(Y)
        return accuracy

class FindingNemo:
    def __init__(self, train_image):
        self.knn = KNN(k=3)
        x_train, y_train = self.convert_image_to_dataset(train_image)
        self.knn.fit(x_train, y_train)

    def convert_image_to_dataset(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        light_orange, dark_orange = (1, 150, 150), (60, 255, 255)
        light_white, dark_white = (0, 0, 150), (145, 60, 255)
        light_black, dark_black = (0, 0, 0), (255, 250, 5)

        orange_mask = cv2.inRange(image_hsv, light_orange, dark_orange)
        white_mask = cv2.inRange(image_hsv, light_white, dark_white)
        black_mask = cv2.inRange(image_hsv, light_black, dark_black)

        final_mask = orange_mask + white_mask + black_mask
        pixels_list_hsv = image_hsv.reshape(-1, 3)

        x_train = pixels_list_hsv / 255
        y_train = final_mask.reshape(-1, ) // 255

        return x_train, y_train

    def remove_background(self, test_image):
        test_image = cv2.resize(test_image, (0, 0), None, .25, .25)
        test_image_hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

        x_test = test_image_hsv.reshape(-1, 3) / 255
        y_pred = self.knn.predict(x_test)

        result = np.array(y_pred).reshape(test_image.shape[:2])
        result = result.astype("uint8")
        final_result = cv2.bitwise_and(test_image, test_image, mask=result)

        return final_result

class FindingDory:
    def __init__(self, train_image):
        self.knn = KNN(k=3)
        x_train, y_train = self.convert_image_to_dataset(train_image)
        self.knn.fit(x_train, y_train)

    def convert_image_to_dataset(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        light_blue, dark_blue = (0, 0, 50), (105, 155, 255)
        light_yellow, dark_yellow = (150, 130, 30), (255, 255, 95)

        mask_blue = cv2.inRange(image_rgb, light_blue, dark_blue)
        mask_yellow = cv2.inRange(image_rgb, light_yellow, dark_yellow)

        final_mask = mask_blue + mask_yellow
        pixels_list_hsv = image_hsv.reshape(-1, 3)

        x_train = pixels_list_hsv / 255
        y_train = final_mask.reshape(-1, ) // 255

        return x_train, y_train

    def remove_background(self, test_image):
        test_image = cv2.resize(test_image, (0, 0), None, .25, .25)
        test_image_hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

        x_test = test_image_hsv.reshape(-1, 3) / 255
        y_pred = self.knn.predict(x_test)

        result = np.array(y_pred).reshape(test_image.shape[:2])
        result = result.astype("uint8")

        final_result = cv2.bitwise_and(test_image, test_image, mask=result)
        final_result = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

        return final_result


if __name__ == "__main__":
    nemo_train_img = cv2.imread("input/nemo.png")
    nemo_test_img = cv2.imread("input/nemo_test.png")

    FN = FindingNemo(nemo_train_img)
    nemo_img_result = FN.remove_background(nemo_test_img)

    plt.imshow(nemo_img_result, "gray")
    plt.show()

    dory_train_img = cv2.imread("input/dory.png")
    dory_test_img = cv2.imread("input/dory_test.png")
    dory_train_img = cv2.resize(dory_train_img, (0, 0), None, .25, .25)

    FD = FindingDory(dory_train_img)
    dory_img_result = FD.remove_background(dory_test_img)

    plt.imshow(dory_img_result, "gray")
    plt.show()
