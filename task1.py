"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np
from collections import deque

def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()



def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character final_cordinates from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be final_cordinates.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the final_cordinates character.
        h: height of the final_cordinates character.
        name: name of character provided or "UNKNOWN".
        Note : the order of final_cordinates characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    enrollment(characters)

    letter_coordinates = detection(test_img)
    
    names = recognition(test_img, letter_coordinates)
    
    results = []
    for i in range(len(letter_coordinates)):
        results.append( {"bbox": letter_coordinates[i], "name": names[i]} )

    # raise NotImplementedError
    return results

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    
    features = {}
    
    
    siftExtractor = cv2.SIFT_create()

    for chs in characters:
        if chs[0]=='dot':
            continue
        _, des = siftExtractor.detectAndCompute(chs[1], None)
        features[chs[0]] = des.tolist()

    with open('features.json', "w") as file:
        json.dump(features, file)

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.

    # implement connected component labeling to detect various candidate characters
    # Total number of characters = 142 + 1 background
    # characters can be between 1/2 and 2x the size of the enrolled images.
    # generate features/ resize to prepare for recogniton

    binary = np.zeros(test_img.shape)
    rows, columns = test_img.shape
    for i in range(rows):
        for j in range(columns):
            if test_img[i][j]<131:
                binary[i][j] = 1 
            else:
                0 
    
    final_cordinates = []
    visited_coordinates = set()

    def DFS(x,y):
        coord_queue = deque()
        coord_queue.append((x,y))
        coordinate_x, coordinate_y, width, height = 10000, 10000, -1000, -1000
        while coord_queue:
            i,j = coord_queue.pop()
            coordinate_x = min(coordinate_x, j)
            coordinate_y = min(coordinate_y, i)
            width = max(j-coordinate_x, width)
            height = max(i-coordinate_y, height)
            visited_coordinates.add((i,j))
            for a,b in [(-1,0), (1,0), (0,1), (0,-1), (-1,-1),(1,1), (-1,1) ,(1,-1)]:
                if 0<=i+a<rows and 0<=j+b<columns and (i+a,j+b) not in visited_coordinates and binary[i+a][j+b]==1: # ask for the image condition
                    coord_queue.append((i+a,j+b))
        
        return [coordinate_x, coordinate_y, width+1, height+1]

    for i in range(rows):
        for j in range(columns):
            if (i,j) not in visited_coordinates and binary[i][j] == 1:
                final_cordinates.append(DFS(i,j))
            else:
                continue

    return final_cordinates

def matcher(dec1, dec2):
    dist = np.zeros((len(dec1),len(dec2)))
    for i in range(dec1.shape[0]):
        for j in range(dec2.shape[0]):             
            d = np.square(dec1[i] - dec2[j])
            ssd = np.sum(d)
            dist[i][j] = ssd
    
    ratio_dist = []

    for i in range(dec1.shape[0]):
        fet_dists = np.argsort(dist[i])             
        best_1,best_2 = fet_dists[0],fet_dists[1]
        if dist[i][best_1] / dist[i][best_2 ] < 0.4:
            ratio_dist.append(best_1)
    
    return ratio_dist

def recognition(test_img, letter_coordinates):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    # using features of the enrollment characters
    # implement matching for all recognition
    
    with open('features.json', "r") as dfile:
        features = json.load(dfile)

    names = []
    siftExtractor = cv2.SIFT_create()

    for bbox in letter_coordinates:
        x,y,w,h = bbox
        image = test_img[y:y+h,x:x+w].astype('uint8')
        key1,dec1 = siftExtractor.detectAndCompute(image, None)

        if len(key1) == 0:
            names.append('dot')
            continue

        key_match = []
        for char in features:
            dec2 = np.array(features[char])
            key_match .append((char,matcher(dec1, dec2)))              # ask for this line 
        key_match .sort(key= lambda x: len(x[1]), reverse=True)

        if len(key_match [0][1]) > 0:
            names.append(key_match [0][0])
        else:
            names.append("UNKNOWN")
    
    return names

    

def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)

if __name__ == "__main__":
    main()