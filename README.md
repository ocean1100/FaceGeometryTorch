## Face Geometry Reconstrucion (PyTorch)

This repository include scripts to
1) Sample and visualize 3D flame models
2) Fit a flame model to an image using 2D landmarks
3) Fit a flame model to a video using 2D landmarks
4) Visualize the results of the video fitting

### Set-up

The has been tested with Python3.7, using pyTorch 1.4.0.
Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh) within the virtual environment.

Make sure your pip version is up-to-date:
```
pip install -U pip
```

Other requirements (including pyTorch) can be installed using:
```
pip install -r requirements.txt
```

The visualization uses OpenGL and vtkplotter which can be installed using:
```
sudo apt-get install python-opengl
pip install -U vtkplotter
```

The Flame model need to be placed under the 'model' folder. It can be download from [MPI-IS/FLAME](http://flame.is.tue.mpg.de/). You need to sign up and agree to the model license for access to the model and the data. We also have it on our drive.


##### Sample FLAME

This demo introduces the different FLAME parameters: shape, expression, neck, jaw.
```
python flame_params_demo.py
```

##### Fit a flame to an image 2D landmarks and project a texture
```
python ./fit_2D_landmarks_to_image.py
Or to change an image
python ./fit_2D_landmarks_to_image.py --target_img_path ./data/bareteeth.000001.26_C.jpg
```

##### Fit a flame to a video (set of images) with 2D landmarks and project their texture
```
python ./fit_multiple_images.py --input_folder PATH-TO-VOCA_PICTURES-SET --output_folder PATH-TO-RESULTS-FOLDER

And to visualize the results simply run:
python ./visualize_video_reconstruction.py --input_folder PATH-TO-VIDEO_RESULTS_FOLDER
```

