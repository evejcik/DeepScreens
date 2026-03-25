# By employing estimated 2D confidence heatmaps of keypoints and an optical-flow consistency constraint, we filter out the
# unreliable estimations of occluded keypoints. When occlusion occurs, we have incomplete 2D keypoints and feed
# them to our 2D and 3D temporal convolutional networks (2D and 3D TCNs) that enforce temporal smoothness to
# produce a complete 3D pose. By using incomplete 2D keypoints, instead of complete but incorrect ones, our networks
# are less affected by the error-prone estimations of occluded keypoints.

# General framework:semi-supervised method
# Human Detection & Keypoints Estimation (done)
# Temporal Convolution for 2D pose
# Temporal Convolution for 3D pose

# 3D pose keypoints are then re-projected to 2D -> these 2D keypoints are used as a form of regularization to the 2D keypoints MSE loss.

# Network 1: outputs the estimated 2D locations of the keypoints for each bounding box of a person -> mmpose already does this for us. 
# Network 2: These maps are combined with optical flow to assess whether the predicted keypoints are occluded. 
# Filter out occluded keypoints and then feed the potentially incomplete 2D keypoints to our second and third networks which are both temporal convolutional networks
# (2D and 3D TCNs respecively) to enforce temporal smoothness.

# The 3D TCN takes potentially incomplete 2D keypoints as input, and requires pairs of a 3D pose and a 2D pose with occlusion labels during training. 
# We also add in a pose regularization term to penalize violations of keypoints that are estimated as unoccluded, contradicting the ground-truth labels.