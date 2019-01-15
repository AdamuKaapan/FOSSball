#!/usr/bin/env python
import numpy as np
import cv2
import collections
from .color_util import *
from .transform_util import *


def erode_and_dilate(img, kernel_size=5, erode_iterations=1, dilate_iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=erode_iterations)
    return cv2.dilate(eroded, kernel, iterations=dilate_iterations)


def find_convex_hull(img):
    contours, h = cv2.findContours(img, 1, 2)
    master_contour = np.concatenate(contours)
    return cv2.convexHull(master_contour)


def approximate_polygon(hull, epsilon_factor=0.1):
    epsilon = epsilon_factor * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    return approx, np.array(approx, dtype="float32")


# Shamelessly stolen from here: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example
def get_warped(img, corner_points, output_size):
    m = cv2.getPerspectiveTransform(corner_points, get_transform_dest_array(output_size))
    return cv2.warpPerspective(img, m, output_size), m


# Averages out the given collection of corner points, useful for smoothing
def calculate_avg_points(deq):
    cum_sum = None

    for i in deq:
        if cum_sum is None:
            cum_sum = i
        else:
            cum_sum += i

    avg = cum_sum / len(deq)
    return np.array(avg, dtype="float32")


def scale_to_match(scale, match, dimension):
    factor = match.shape[dimension] / scale.shape[dimension]
    return cv2.resize(scale, (0, 0), fx=factor, fy=factor)


def get_field_img(img, corners):
    # Get the green parts only
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))

    # Remove unnecessary/noisy small results
    filtered = erode_and_dilate(mask)

    # Find the convex hull representing the filtered area
    hull = find_convex_hull(filtered)

    # Approximate a polygon from the convex hull
    approx, src_float_array = approximate_polygon(hull)
    # cv2.drawContours(img, [approx], 0, (255, 0, 255), 2)

    sorted_float_array = sort_source_float_array(src_float_array)

    corners.append(sorted_float_array.copy())
    avg_float_array = calculate_avg_points(corners)

    regulation_foosball_ratio = 56 / 30  # 56 inches wide by 30 inches deep
    img_width = img.shape[1]
    max_height = int(img_width / regulation_foosball_ratio)
    max_width = int(img_width)

    return get_warped(img, avg_float_array, (max_width, max_height))


def main():
    cap = cv2.VideoCapture('zacs_foosball.mov')
    writer = cv2.VideoWriter('output.wmv', cv2.VideoWriter_fourcc(*'WMV2'), 60.0, (1920, 1028))

    # Increasing maxlen makes the motion smoother at the cost of slower response to camera position changes
    corners = collections.deque([], maxlen=60)

    ball_points = collections.deque([], maxlen=120)

    current_frame = -1

    field_width = 56  # inches
    field_height = 30  # inches
    framerate = 240

    while True:
        ret, img = cap.read()

        if not ret:
            break

        current_frame += 1

        if current_frame < 1600:
            continue

        warped, transform = get_field_img(img, corners)

        x_pixels_per_inch = warped.shape[1] / field_width
        y_pixels_per_inch = warped.shape[0] / field_height

        _, inverted = cv2.invert(transform)

        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        sat_min = 15
        sat_max = 255
        val_min = 15
        val_max = 255

        mask1 = cv2.inRange(hsv, np.array([0, sat_min, val_min]), np.array([15, sat_max, val_max]))
        mask2 = cv2.inRange(hsv, np.array([115, sat_min, val_min]), np.array([180, sat_max, val_max]))
        mask = cv2.bitwise_or(mask1, mask2)

        filtered = erode_and_dilate(mask, dilate_iterations=2, erode_iterations=2)

        contours, h = cv2.findContours(filtered.copy(), 1, 2)

        ball_found = False

        for c in contours:
            area = cv2.contourArea(c)

            if not 1000 < area < 3000:
                continue

            perimeter = cv2.arcLength(c, True)

            if perimeter == 0:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            max_circularity_ratio = 1.4

            if (np.pi * radius * radius) / area < max_circularity_ratio:
                moments = cv2.moments(c)
                c_x = int(moments["m10"] / moments["m00"])
                c_y = int(moments["m01"] / moments["m00"])

                ball_found = True
                ball_points.append((c_x, c_y))

                # cv2.putText(warped, str(area), (c_x, c_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                break

        if ball_found:
            pass

        for index, point in enumerate(ball_points):
            lerp_val = index / len(ball_points)
            rgb = hsv_to_rgb((lerp_val, 1.0, 1.0))
            cv2.circle(warped, point, 6, rgb, -1)
            # Draw inverted on the original image
            # transformed = transform_coord(point, inverted)
            # cv2.circle(img, transformed, 6, rgb, -1)

        # Draw over most recent point
        cv2.circle(warped, ball_points[-1], 12, (0, 0, 255), -1)

        cv2.imshow('out', cv2.resize(warped, (0, 0), fx=0.5, fy=0.5))
        #cv2.imshow('img', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))

        # Output Processing
        # scaled_img = scale_to_match(img, warped, 1)
        #
        # combined = np.vstack((warped, scaled_img))
        # combined = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5)
        #
        # cv2.imshow('out', combined)
        #
        # writer.write(warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
