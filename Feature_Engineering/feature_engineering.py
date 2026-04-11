#input: annotated csv 
#output: 



import pandaas as pd
import numpy as np

#1. encode reliability category as integer: trust = 0, partial trust = 1, don't trust = 2
#2. encode joint_name as integer (0 - 16), using H36M mapping from mmpose
#3. handle missing values for 'valid' -> if  