### Data preparation
To train the CNN models, you need three kinds of data: raw images, hand masks and object/hand heatmaps. See examples in `$root/raw_data`.

1. Raw images: capture about 1000 images while performing different actions. For better hand segmentation, capture data from different subjects.

2. Hand masks: you can manually annoate hand regions or use background subtraction to generate hand masks for the raw images. 

3. Heatmaps: the object/hand heatmap contains probability map for object/hand locations encoded in three channels: blue for object of interest, green for left hand, red for right hand. `$root/annotation` contains a few tools to label object and hands locations and generate heatmaps.

### Hand segmentation training
To train the hand segmentation network, go to `$root/segmentation_training`.

1. Get Caffe from [here](https://github.com/minghuam/caffe-dev) and modify the `caffe_root` in `config.py`.

2. If the Caffe is built successfully and the Caffe root path is correct, you can run the demo and see results like the following:

        python demo.py

    ![Alt text](https://github.com/minghuam/HODM/blob/master/segmentation_training/demo_data/demo.png)

3. Run the following command to resize raw training data and generate training data source text file: `training_data.txt`.
        
        python prepare_training_data.py

4. Train the hand segmentation CNN. The trained model files are saved in `$root/segmentation_training/model`.

        sh train.sh
        

### Hands/Object detection training

1. Set `caffe_root` in `config.py` to the same Caffe version. Set `hand_model` to the path of the trained hand segmentation model.

2. If the configuration is correct, you can run the demo for hand and object detection:
        
        python demo.py

    ![Alt text](https://github.com/minghuam/HODM/blob/master/detection_training/demo_data/demo.png)
        
3. Run the following command to resize raw training data and generate training data source text file: `training_data.txt`.
        
        python prepare_training_data.py

4. Train the hand/object detection CNN. The trained model files are saved in `$root/detection_training/model`.

        sh train.sh
