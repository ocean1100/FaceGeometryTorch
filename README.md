## FLAME: Faces Learned with an Articulated Model and Expressions (TF)

[FLAME](http://flame.is.tue.mpg.de/) is a lightweight and expressive generic head model learned from over 33,000 of accurately aligned 3D scans. This repository provides sample Tensorflow code to experiment with the FLAME model. Parts of the repository are adapted from the [Chumpy](https://github.com/mattloper/chumpy)-based [FLAME-fitting repository](https://github.com/Rubikplayer/flame-fitting). 

<p align="center"> 
<img src="gifs/model_variations.gif">
</p>

FLAME combines a linear identity shape space (trained from 3800 scans of human heads) with an articulated neck, jaw, and eyeballs, pose-dependent corrective blendshapes, and additional global expression blendshapes. For details please about the model, please see the [scientific publication](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf) and the [supplementary video](https://youtu.be/36rPTkhiJTM).

### Content

This repository include scripts to
1) Sample and visualize 3D flame models
2) Fit a flame model to an image using 2D landmarks

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
python ./fit_2D_landmarks_to_image.py
