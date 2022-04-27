# Image-Processing-Self-Driving-Cars
### Educational Project 2022

### Phase 1 - Lane Line detection:
Expected output:
For the first phase, the expected output is as follows:
- Your pipeline should be able to detect the lanes, highlight them using a fixed color, and pain the area between them in any color you like (it’s painted green in the image above.)
- You’re required to be able to roughly estimate the vehicle position away from the center of the lane.
- As a bonus, you can try to estimate the radius of curvature of the road as well.


### Approach Steps:
- Read photo
- Apply Pre-processing Techniques
- Try getting HLS channels first to get rid of lightness noise
- Extract edges with Sobel from different informative views and stacking them together
- Try to apply perspective transform to convert to Bird's eye view
- Need to impelement a function to detect lane line using sliding window
- Why not improve the prvious function instead of performing blind search with each new frame, it should use info extracted from the previous frame
- Calculate radius of curvature and position from the center of the lane
- Got all final outputs and implemented stacked image to use in debugging mode
- Experiment with video


### Phase 2 - Locate and Identify cars on the road


### Usage guide:
- we made a dynamic way to run our function taking one of the following options

1) The video name 
2) The video name and and the stacked view boolean
3) All the parameters(which are video name, stacked view, resize param, view, save)

- in 1 and 2, the other paramaters are set to their default values.
