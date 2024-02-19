#!/usr/bin/env python

import rospy
import tf2_ros
import numpy as np
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray

ARM_JOINTS_NAMES = ["FrontalElevationFlexion", "FrontalElevationExtension", "Abduction","ElbowPronosupination","ElbowFlexion"]
TORSO_TF_NAME = "C7"

class OcraAngles():
    def __init__(self):
        jointMessage = JointState()
        jointMessage.name = ARM_JOINTS_NAMES
        jointMessage.header.stamp = rospy.Time.now()
        jointMessage.position = None
        jointMessage.velocity = None
        jointMessage.effort   = None

        self.right_arm = jointMessage
        self.left_arm  = jointMessage

        self.torso = None
    
        rospy.Subscriber("/jointLimb", JointState, self.callbackLimb)
        self.right_arm_pub = rospy.Publisher('ocra/right_arm_angles', JointState, queue_size=10)
        self.left_arm_pub = rospy.Publisher('ocra/left_arm_angles', JointState, queue_size=10)
        self.torso_pub = rospy.Publisher('ocra/torso', JointState, queue_size=10)

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

    def publishAngles(self):
        if self.right_arm.position:
            self.right_arm_pub.publish(self.right_arm)
            self.left_arm_pub.publish(self.left_arm)
        
        if self.torso != None:
            self.torso_pub.publish(self.torso)
    
    def unit_vector(v):
        return (v / np.linalg.norm(v))

    def project_onto_plane(self, v1, n):
        # Subtract from vector v1 its component along n, the normal vector to the plane
        prj_n = np.dot(v1, self.unit_vector(n))
        return [v1-prj_n]
    
    def angle_between_vectors(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u)),-1.0, 1.0) 

    def computeTorso(self):
        #Get Torso frame
        try:
            tf = self.tfBuffer.lookup_transform(TORSO_TF_NAME, 'world', rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        
        # Get vertical axis (y)
        quat = tf.rotation
        matrix = R.from_quat([quat.x,quat.y,quat.z,quat.w]).as_matrix()
        y = matrix[:,1]

        y = np.vectorize(y)
        n = np.array([1,0,0])
        z = np.array([0,0,1])
        y_prj = self.project_onto_plane(self,y,n)

        self.torso = self.angle_between_vectors(self,y_prj,z)

    def callbackLimb(self,arm):
            right_arm = False
            left_arm = False

            if "RightArm" in arm.name[0]:
                right_arm = True
            elif "LeftArm" in arm.name[0]:
                left_arm = True
            else:
                return 
            
            q1 = arm.position[0]
            if q1>=0:
                frontal_elevation_flexion = q1
                frontal_elevation_extension = 0
            else:
                frontal_elevation_flexion = 0
                frontal_elevation_extension = -q1

            abduction = arm.position[1] #q2
            elbow_pronosupination = np.abs(arm.position[2]) #q3
            elbow_flexion = np.abs(arm.position[3]) #q4

            position = [frontal_elevation_flexion,frontal_elevation_extension,abduction,elbow_pronosupination,elbow_flexion]

            if right_arm:
                self.right_arm.positio = position
                self.right_arm.header = arm.header
            elif left_arm:
                self.left_arm.positio = position
                self.left_arm.header = arm.header
            else:
                raise Exception("Error in the arm's joints message")


if __name__ == '__main__':
    rospy.init_node('ocra_angles', anonymous=True)
    ocra_angles = OcraAngles()

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rospy.spin()
        ocra_angles.computeTorso()
        ocra_angles.publishAngles()
        rate.sleep()

