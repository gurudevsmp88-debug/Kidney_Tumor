"""panolidar: LiDAR-grade metric 3D from one or two equirectangular panoramas.

Core idea: fuse a learned panoramic depth prior with two-panorama spherical
triangulation, weighting each ray by its *analytic* triangulation uncertainty,
which blows up near the epipoles (the direction of travel). A dense city
capture supplies uncertainty-weighted teacher geometry for training.
"""
