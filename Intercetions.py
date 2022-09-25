import math
import numpy as np
import csv
from numpy.core import _methods
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from geometric_median import *


def get_intercetions(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return (x3, y3, x4, y4)


def print_image():
    return True


def get_intersection_circles(r_a, r_b, r_c, r_d, r_e, str_real_position):
    # intersection circles
    x_a, y_a = 10, 135
    x_b, y_b = 290, 390
    x_c, y_c = 10, 460
    x_d, y_d = 520, 500
    x_e, y_e = 420, 130

    if print_image():
        circle_a = plt.Circle((x_a, y_a), r_a, color='b', fill=False)
        circle_b = plt.Circle((x_b, y_b), r_b, color='b', fill=False)
        circle_c = plt.Circle((x_c, y_c), r_c, color='b', fill=False)
        circle_d = plt.Circle((x_d, y_d), r_d, color='b', fill=False)
        circle_e = plt.Circle((x_e, y_e), r_e, color='b', fill=False)

        fig, ax = plt.subplots()
        ax.set_xlim((-500, 1000))
        ax.set_ylim((-500, 1000))
        ax.add_artist(circle_a)
        ax.add_artist(circle_b)
        ax.add_artist(circle_c)
        ax.add_artist(circle_d)
        ax.add_artist(circle_e)

    all_intersections = list()

    intersections = get_intercetions(x_a, y_a, r_a, x_b, y_b, r_b)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    intersections = get_intercetions(x_a, y_a, r_a, x_c, y_c, r_c)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    intersections = get_intercetions(x_a, y_a, r_a, x_d, y_d, r_d)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    intersections = get_intercetions(x_a, y_a, r_a, x_e, y_e, r_e)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    intersections = get_intercetions(x_b, y_b, r_b, x_c, y_c, r_c)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    intersections = get_intercetions(x_b, y_b, r_b, x_d, y_d, r_d)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    intersections = get_intercetions(x_b, y_b, r_b, x_e, y_e, r_e)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    intersections = get_intercetions(x_c, y_c, r_c, x_d, y_d, r_d)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    intersections = get_intercetions(x_c, y_c, r_c, x_e, y_e, r_e)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    intersections = get_intercetions(x_d, y_d, r_d, x_e, y_e, r_e)
    if intersections is not None:
        all_intersections.append(intersections)
        i_x3, i_y3, i_x4, i_y4 = intersections
        if print_image():
            plt.plot([i_x3, i_x4], [i_y3, i_y4], '.', color='r')

    if print_image():
        plt.gca().set_aspect('equal', adjustable='box')

    # print(all_intersections)

    intersection_points = list()

    for intersection in all_intersections:
        point1 = (intersection[0], intersection[1])
        point2 = (intersection[2], intersection[3])
        intersection_points.append(point1)
        intersection_points.append(point2)

    a = np.array(intersection_points)

    # print(a)

    str_real_position = str_real_position.replace('[', '')
    str_real_position = str_real_position.replace(']', '')
    test = str_real_position.split(',')
    real_position = []
    for i in range(len(test)):
        real_position.append(int(test[i]))

    position = geometric_median(a, method='minimize')
    distance = math.sqrt(((real_position[0] - position[0]) ** 2) + ((real_position[1] - position[1]) ** 2))
    print(real_position, position, distance)

    if print_image():
        plt.plot([position[0]], [position[1]], '+', color='g')
        plt.plot([real_position[0]], [real_position[1]], '+', color='c')
        title = "position: "+str(int(position[0]))+" "+str(int(position[1]))+\
                ", realPosition: "+str(int(real_position[0]))+" "+str(int(real_position[1]))+\
                ", distance: "+str(int(distance))

        plt.title(title)
        if 100 < distance > 150:
            print("print")
            plt.show()

    plt.close()
    return distance


with open('predictions_file.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    distances = []
    for row in csv_reader:
        distances.append(get_intersection_circles(int(float(row[0])), int(float(row[1])), int(float(row[2])),
                                                  int(float(row[3])), int(float(row[4])), row[5]))

print(distances)
print(sum(distances) / len(distances))

_25 = 0
_50 = 0
_75 = 0
_100 = 0
_125 = 0
_150 = 0
far = 0

for i in range(len(distances)):
    if distances[i] <= 25:
        _25 = _25 + 1

    elif 25 < distances[i] <= 50:
        _50 = _50 + 1

    elif 50 < distances[i] <= 75:
        _75 = _75 + 1

    elif 75 < distances[i] <= 100:
        _100 = _100 + 1

    elif 100 < distances[i] <= 125:
        _125 = _125 + 1

    elif 125 < distances[i] <= 150:
        _150 = _150 + 1

    else:
        far = far + 1

print("25:", _25, ", 50:", _50, ", 75:", _75, ", 100:", _100, ", 125:", _125, ", 150:", _150, ", far:", far)

plt.hist(distances, bins=16)
plt.xlabel("Prediction Error [Distance]")
_ = plt.ylabel("Count")
plt.show()
