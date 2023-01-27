## Animation Research Utils
A collection of useful python utils for machine learning research including readinf and saving animation (.bvh) files,
a task agnostic animation representation, functions for common animation data manipulation; all suitable for
training machine learning models.

Files ending with _ts are pytorch equivalents enabling backprop.

To run tests (by running each python script directly) ensure you modify TEST_FILEPATH in test_config.py
to point to a valid bvh file on your filesystem. 
If you're wondering where to get some, try [this](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-motionbuilder-friendly-bvh-conversion-release-of-cmus-motion-capture-database).

### Dependencies
- numpy
- (optional but recommended) plotly for use of plot.py
- (optional) pytorch to use functions in *_ts.py files