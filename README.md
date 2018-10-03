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

* Create a new virtual environment with Python 3

`virtualenv -p python3 ~/Environments/ShazamMovies`

* activate the virtual environment

`source ~/Environments/ShazamMovies/bin/activate`

* Install OpenCV for Python and dependencies e.g. numpy

`pip install -r requirements.txt`
