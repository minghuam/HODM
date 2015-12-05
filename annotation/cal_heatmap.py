import os,sys
import cv2
import numpy as np
import scipy

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = scipy.mgrid[-size:size+1, -sizey:sizey+1]
    g = scipy.exp(-(x**2/float(size*8)+y**2/float(sizey*8)))
    return g / g.max()

Ig_hand = (gauss_kern(200)*255).astype(np.uint8)
Ig_obj = (gauss_kern(500)*255).astype(np.uint8)

images_dir = '../raw_data/img'
output_dir = '../raw_data/heatmap'

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

hands = dict()
objs = dict()
with open('hands.txt', 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		img = tokens[0]
		x1 = int(tokens[1])
		y1 = int(tokens[2])
		x2 = int(tokens[3])
		y2 = int(tokens[4])
		hands[img] = ((x1,y1), (x2,y2))

with open('objects.txt', 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		img = tokens[0]
		x1 = int(tokens[1])
		y1 = int(tokens[2])
		objs[img] = ((x1,y1),)

for img in hands:
	p_obj = objs[img][0]
	(p_left, p_right) = hands[img]

	I = cv2.imread(os.path.join(images_dir, img))
	Ih = np.zeros(I.shape, np.uint8)

	if p_obj[0] != -1:
		cv2.circle(Ih, p_obj, 5, (255, 0, 0), -1)
	if p_left[0] != -1:
		cv2.circle(Ih, p_left, 5, (0, 255, 0), -1)
	if p_right[0] != -1:
		cv2.circle(Ih, p_right, 5, (0, 0, 255), -1)

	Ih_c = np.zeros(I.shape, np.uint8)

	k_size = (Ig_obj.shape[0] - 1)/2
	Ih_p = np.zeros((I.shape[0] + Ig_obj.shape[0]*2, I.shape[1] + Ig_obj.shape[0]*2), np.uint8)
	if p_obj[0] != -1:
		x1_p = p_obj[0] + Ig_obj.shape[1]
		y1_p = p_obj[1] + Ig_obj.shape[0]
		Ih_p[y1_p-k_size:y1_p+k_size+1, x1_p-k_size:x1_p+k_size+1] = Ig_obj
	Ih_c[:,:,0] = Ih_p[Ig_obj.shape[0]:Ig_obj.shape[0]+I.shape[0], Ig_obj.shape[1]:Ig_obj.shape[1]+I.shape[1]]	

	k_size = (Ig_hand.shape[0] - 1)/2
	Ih_p = np.zeros((I.shape[0] + Ig_hand.shape[0]*2, I.shape[1] + Ig_hand.shape[0]*2, 2), np.uint8)
	if p_left[0] != -1:
		x1_p = p_left[0] + Ig_hand.shape[1]
		y1_p = p_left[1] + Ig_hand.shape[0]
		Ih_p[y1_p-k_size:y1_p+k_size+1, x1_p-k_size:x1_p+k_size+1, 0] = Ig_hand
	if p_right[0] != -1:
		x2_p = p_right[0] + Ig_hand.shape[1]
		y2_p = p_right[1] + Ig_hand.shape[0]
		Ih_p[y2_p-k_size:y2_p+k_size+1, x2_p-k_size:x2_p+k_size+1, 1] = Ig_hand
	Ih_c[:,:,1:] = Ih_p[Ig_hand.shape[0]:Ig_hand.shape[0]+I.shape[0], Ig_hand.shape[1]:Ig_hand.shape[1]+I.shape[1], :]

	cv2.imwrite(os.path.join(output_dir, img), Ih_c)

	I = cv2.addWeighted(I, 0.75, Ih_c, 0.5, 0)
	cv2.imshow('I', I)
	if cv2.waitKey(10) & 0xFF == 27:
		break