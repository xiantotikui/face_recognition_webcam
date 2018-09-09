import os
import random
import cv2
import numpy as np

class TripletGenerator:
    def __init__(self):
        self

    def generator_triplet_loss(self, examples, input):
        a_array = []
        p_array = []
        n_array = []
        files = iter(os.listdir(input))
        y_array = []
        for i in range(examples):
            a = next(files)
            n = random.choice(os.listdir(input))
            while a == n:
                n = random.choice(os.listdir(input))

            a_path = os.path.join(input, a)
            n_path = os.path.join(input, n)

            a_file = random.choice(os.listdir(a_path))
            p_file = random.choice(os.listdir(a_path))
            while a_file == p_file:
                p_file = random.choice(os.listdir(a_path))

            n_file = random.choice(os.listdir(n_path))

            a_img = cv2.imread(os.path.join(a_path, a_file))
            p_img = cv2.imread(os.path.join(a_path, p_file))
            n_img = cv2.imread(os.path.join(n_path, n_file))

            a_img = np.flip(a_img, 1) / 255
            p_img = np.flip(p_img, 1) / 255
            n_img = np.flip(n_img, 1) / 255

            a_array.append(a_img)
            p_array.append(p_img)
            n_array.append(n_img)
            y_array.append(a)
            
            print(i)

        return(a_array, p_array, n_array, y_array)

    def generator_from_webcam(self, examples, input, video):
        a_array = []
        p_array = []
        n_array = []
        files = iter(os.listdir(input))
        for i in range(examples):
            n = random.choice(os.listdir(input))
            n_path = os.path.join(input, n)
            n_file = random.choice(os.listdir(n_path))
            n_img = cv2.imread(os.path.join(n_path, n_file))

            a_img = np.flip(random.choice(video), 1) / 255
            p_img = np.flip(random.choice(video), 1) / 255
            
            while (a_img[0][0][0] == p_img[0][0][0]) or (a_img[-1][-1][-1] == p_img[-1][-1][-1]):
                p_img = random.choice(video)
            
            n_img = np.flip(n_img, 1) / 255

            a_array.append(a_img)
            p_array.append(p_img)
            n_array.append(n_img)

        return(a_array, p_array, n_array)

