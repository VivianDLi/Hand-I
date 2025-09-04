"""
DVS Dataset plan:

1. Real-world DVS from paired DVS-tiff data (relax21)
    a. Obtain ground truth data by running MediaPipe on tiff
    b. Correlate with dvs .npy data
2. Simulated DVS data
    a. Generate random hand pose (random hand sizes and rotations)
    b. Create image from random viewing direction
    c. Add random noise to background of image

Data Parameters:
    a. Hand Marker Positions (21) - see MediaPipe (i.e., positions in image from 0.0-1.0)
    b. Hand Marker World Positions (21) - positions in 3D space using the wrist point as the origin
    b. Hand Location (2) - bounding box for hand location
    c. Hand Present (1) - binary representing confidence
    d. Handedness (1) - left or right hand
    
Models:
    a. Hand Predictor - uses full image to generate hand location bounding box (c)
    b. Hand Landmark Predictor - uses cropped image of just hand to generate (a, b, d, e)
"""

def generate_synthetic_dvs_hand():
    pass