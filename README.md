# DiscTracker

## Background
 
In indoor ultimate frisbee, each point begins with one team throwing the disc to the other team on opposing sides of the field, called a "pull".
For this pull to be deemed "valid", it must pass through any part of an imaginary two-meter-high box bounded by the front, back and sidelines of the receiving team's endzone. 
The dimensions of the pitch in the demo footage provided are shown in Figure 2.
By tracking the position of the disc in 3-dimensional space it is possible to determine if its path intersects this box, and thus whether the pull was valid. 
Figure 1 shows a representation of the pitch used, in 3-dimensional space, with the two-meter-high bounding boxes plotted.

![FrisbeePitch](https://github.com/JoelM935/DiscTracker/assets/33060876/cc745ae0-dd6c-4a7f-9cb0-d1b17ad47212)\
*Fig. 1 3D representation of an indoor ultimate pitch, with the 2m high end zone boxes drawn*

## Method
### Camera Setup
![Camera_Setup](https://github.com/JoelM935/DiscTracker/assets/33060876/03c96ae6-fb88-4036-beb6-71389b3777eb)\
*Fig. 2 Setup used to capture footage* 

This program takes two input videos which must be from two parallel-facing cameras (separated by some distance $d$) and timesynced. A diagram of the setup for the demo footage is shown in Figure 2. Any new footage must follow the same setup (though exact pitch dimensions can change).

### Object detection and tracking
For each video, a series of steps are run to detect and track objects within the footage.

1. A Gaussian mixture model (GMM) is applied to create a binary mask of the foreground pixels.
2. Morphological processing is used to remove noise and stabilize detections identified using the GMM.
3. Blob analysis is used to identify groups of connected white (foreground) pixels and record their positions.
4. For the first frame all objects are given an ID. For all subsequent frames, we calculate the distance between all of the object positions from the previous frame and those detected in the current frame. We use the resulting matrix as the cost matrix and assign objects from the current frame an ID using the Jonker-Volgenant algorithm. In short, we update the position of an ID with the closest object to the ID's last position. Any leftover objects are given a new ID.

### De-projection
Having run the above process for both videos, we now have a series of pixel coordinates for each object over time. Using Figure 2, by the construction of similar triangles it can be shown that,
$$\frac{x_L}{f}=\frac{X+\frac{d}{2}}{Y}$$
$$\frac{x_R}{f}=\frac{X-\frac{d}{2}}{Y}$$
$$\frac{z_L}{f}=\frac{z_R}{f}=\frac{Z}{Y},$$
where $(x_i, z_i)$ are the pixel coordinates of the object in camera $i$, $d$ is the separation between the two cameras, $f$ is the focal length of the camera (converted into pixels) and $(X, Y, Z)$ are the coordinates of the object in 3D space.

By solving these equations for $(X, Y, Z)$ we get,
$$X=\frac{d(x_L+x_R)}{2(x_L-x_R)}$$
$$Z=\frac{d(z_L+z_R)}{2(x_L-x_R)}$$
$$Y=\frac{df}{(x_L-x_R)}.$$

N.B Strictly $Z=\frac{dz_L}{x_L-x_R}=\frac{dz_R}{x_L-x_R}$, but as our setup won't be perfect we take the average of both $Z$ coordinates.

## Installation
This package can be installed using:
```bash
pip install git+https://github.com/JoelM935/DiscTracker@main
```
The example footage folder in the `data` directory can also be downloaded from GitHub.

## Running the program
Once installed, a command line executable `disc_tracker` will be available on your machine.

Before running the program, the input data must be packaged up in the following format:
```
└── 📁name                      <- Name for the dataset
    └── camera_settings.yaml    <- Yaml file containing camera settings
    └── pitch_dimensions.yaml   <- Yaml file containing pitch dimensions (optional)
    └── 📁tracks                <- Output directory for track data
    └── 📁video
        └── left.mp4            <- Synced video from the left camera
        └── right.mp4           <- Synced video from the right camera
```
The `camera_settings` yaml, must be formatted as follows:
```yaml
resolution: [x, y]  # Video resolution in pixels
sensor_width: <width of the camera sensor in meters>
focal_length: <focal length of the lens in meters>
d: <horizontal separation of the cameras in meters>
c: <distance between cameras and back of endzone in meters>
h: <height of cameras above ground in meters>
```
The optional `pitch_dimensions` yaml, must be formatted as follows:
```yaml
width: <width of pitch in meters>
length: <length of pitch in meters>
endzone_depth: <depth of end zones in meters>
```
if no `pitch_dimensions` yaml is present, the default UKU measurements will be used (40m x 20m with 5m endzones).

Once this is set up, the program can be run using:
```bash
disc_tracker /path/to/dataset/
```
This will plot the result using Plotly by default. To plot using Matplotlib use:
```bash
disc_tracker /path/to/dataset/ -p mpl
```
If track data already exists in the `tracks` sub-directory, they can be plotted using the `-plot_only` option.
 
