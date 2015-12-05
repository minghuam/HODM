#!/bin/bash

# raw data
# https://drive.google.com/open?id=0B6HlWbKjTllQN012dkNtZHJhb3c
wget -O raw_data.tar.gz --no-check-certificate https://googledrive.com/host/0B6HlWbKjTllQN012dkNtZHJhb3c
tar -xvzf raw_data.tar.gz

# segmentation test video
# https://drive.google.com/open?id=0B6HlWbKjTllQYm1JS3ExSWhhTlk
wget -O segmentation_training/demo_data/test.avi --no-check-certificate https://googledrive.com/host/0B6HlWbKjTllQYm1JS3ExSWhhTlk

# segmentation demo model
# https://drive.google.com/open?id=0B6HlWbKjTllQZDhScHRuWDlEWlk
wget -O segmentation_training/demo_data/HAND_iter_3000.caffemodel --no-check-certificate https://googledrive.com/host/0B6HlWbKjTllQZDhScHRuWDlEWlk

# segmentation pascal model
# https://drive.google.com/open?id=0B6HlWbKjTllQZ1VTRnJ0SDgxdU0
wget -O segmentation_training/fcn-32s-pascalcontext.caffemodel --no-check-certificate https://googledrive.com/host/0B6HlWbKjTllQZ1VTRnJ0SDgxdU0


# detection test video
# https://drive.google.com/open?id=0B6HlWbKjTllQQU4xaWpmMnJPRnc
wget -O detection_training/demo_data/test.avi --no-check-certificate https://googledrive.com/host/0B6HlWbKjTllQQU4xaWpmMnJPRnc

# detection demo model
# https://drive.google.com/open?id=0B6HlWbKjTllQWnkzWldiNnJvbG8
wget -O detection_training/demo_data/OBJ_iter_4000.caffemodel --no-check-certificate https://googledrive.com/host/0B6HlWbKjTllQWnkzWldiNnJvbG8
