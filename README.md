# ShazamMovies

* Prerequesites to install
  * Compiler
  
    `sudo apt-get install build-essential`

  * Required
  
    `sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev`

  * Optional
  
    `sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev`

* Clone the project
```
cd ~/Projects
git clone https://github.com/Adamouization/ShazamMovies
cd ShazamMovies
```

* Create a new virtual environment

`virtualenv ~/Environments/ShazamVideo`

* activate the virtual environment

`source ~/Environments/ShazamVideo/bin/activate`

* Install OpenCV for Python and other libraries

`pip install opencv-python`
