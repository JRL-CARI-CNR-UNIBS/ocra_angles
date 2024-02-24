#! /usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R

# Camera namespace
CAMERA = "camera"

# Tf
TORSO_TF_NAME = CAMERA+"/neck"
LEFT_SHOULDER_TF_NAME = CAMERA+"/left_shoulder"
RIGHT_SHOULDER_TF_NAME = CAMERA+"/right_shoulder"
LEFT_ELBOW_TF_NAME = CAMERA+"/left_shoulder_elbow"
RIGHT_ELBOW_TF_NAME = CAMERA+"/right_shoulder_elbow"
LEFT_WRIST_TF_NAME = CAMERA+"/left_shoulder_wrist"
RIGHT_WRIST_TF_NAME = CAMERA+"/right_shoulder_wrist"

#Listen from
# LIMBS_TOPIC = "/"+CAMERA+"/limb_joint"

#Publish on
RIGHT_ARM_TOPIC = "ocra/right_arm_angles"
LEFT_ARM_TOPIC = "ocra/left_arm_angles"
TORSO_TOPIC = "ocra/torso"
LEFT_ARM_JOINT_PREFIX = "left_arm"
RIGHT_ARM_JOINT_PREFIX = "right_arm"

#Angles names
ARM_JOINTS_NAMES = ["Frontal", "Lateral","Rotation","Flexion"]

class OcraAngles():
    def __init__(self):
        jointMessage = JointState()
        jointMessage.name = ARM_JOINTS_NAMES
        jointMessage.header.stamp = rospy.Time.now()
        jointMessage.position = None
        jointMessage.velocity = [0.0,0.0,0.0,0.0] #does not matter
        jointMessage.effort   = [0.0,0.0,0.0,0.0] #does not matter

        self.right_arm = jointMessage
        self.left_arm  = jointMessage

        self.torso = None
    
        #rospy.Subscriber(LIMBS_TOPIC, JointState, self.callbackLimb)
        self.right_arm_pub = rospy.Publisher(RIGHT_ARM_TOPIC, JointState, queue_size=10)
        self.left_arm_pub = rospy.Publisher(LEFT_ARM_TOPIC, JointState, queue_size=10)
        self.torso_pub = rospy.Publisher(TORSO_TOPIC, Float64, queue_size=10)

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

    # def publishAngles(self):
    #     if self.right_arm.position != None:
    #         self.right_arm_pub.publish(self.right_arm)
    #         self.left_arm_pub.publish(self.left_arm)
        
    #     if self.torso != None:
    #         self.torso_pub.publish(self.torso)

    #     self.right_arm.position = None
    #     self.left_arm.position = None
    #     self.torso = None

    def computeAngles(self):
        try:
            tf_left_elbow = self.tfBuffer.lookup_transform(LEFT_SHOULDER_TF_NAME,LEFT_ELBOW_TF_NAME,rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            tf_left_elbow = None

        try:
            tf_right_elbow = self.tfBuffer.lookup_transform(RIGHT_SHOULDER_TF_NAME,RIGHT_ELBOW_TF_NAME,rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            tf_right_elbow = None

        try:
            tf_left_wrist = self.tfBuffer.lookup_transform(LEFT_SHOULDER_TF_NAME,LEFT_WRIST_TF_NAME,rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            tf_left_wrist = None

        try:
            tf_right_wrist = self.tfBuffer.lookup_transform(RIGHT_SHOULDER_TF_NAME,RIGHT_WRIST_TF_NAME,rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            tf_right_wrist = None

        # Left arm
        if tf_left_elbow != None and tf_left_wrist != None:
            left_frontal = self.computeFrontal(tf_left_elbow)
            left_lateral = self.computeLateral(tf_left_elbow)
            left_rotation = self.computeRotation(tf_left_wrist)
            left_flexion = self.computeFlexion(tf_left_wrist)
            left_arm = [left_frontal,left_lateral,left_rotation,left_flexion]
            self.left_arm.position = left_arm
        else:
            self.left_arm.position = None

        # Right arm
        if tf_right_elbow != None and tf_right_wrist != None:
            right_frontal = -self.computeFrontal(tf_right_elbow)
            right_lateral = self.computeLateral(tf_right_elbow)
            right_rotation = self.computeRotation(tf_right_wrist)
            right_flexion = self.computeFlexion(tf_right_wrist)
            right_arm = [right_frontal,right_lateral,right_rotation,right_flexion]
            self.right_arm.position = right_arm
        else:
            self.right_arm.position = None
        
        # Torso
        self.computeTorso()

        # Publish
        if self.left_arm.position != None:
            self.left_arm_pub.publish(self.left_arm)

        if self.right_arm.position != None:
            self.right_arm_pub.publish(self.right_arm)
        
        if self.torso != None:
            self.torso_pub.publish(self.torso)
    
    def unit_vector(self,v):
        return (v / np.linalg.norm(v))

    def project_onto_plane(self, v1, n):
        # Subtract from vector v1 its component along n, the normal vector to the plane
        unit_n = self.unit_vector(n)      
        prj_n = np.dot(v1, unit_n)*unit_n
        res = v1-prj_n

        # print("n: ",unit_n, "y: ",v1)
        # print("projection on n: ",prj_n)
        # print("res: ",res)

        return res
    
    def angle_between_vectors(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u),-1.0, 1.0))

    def computeTorso(self):
        #Get Torso frame
        try:
            tf = self.tfBuffer.lookup_transform('world',TORSO_TF_NAME,rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None
        
        # print("Torso frame found->\n",tf)

        # Get vertical axis (y)
        quat = tf.transform.rotation
        matrix = R.from_quat([quat.x,quat.y,quat.z,quat.w]).as_matrix()
        y = matrix[:,1]

        y = np.array(y)
        n = np.array([1,0,0])
        z = np.array([0,0,1])

        y_prj = self.project_onto_plane(y,n)

        self.torso = self.angle_between_vectors(y_prj,z)*180/np.pi
        
        # print("Torso angle: ",self.torso)
        # print("-------------------")

    # def callbackLimb(self,arm):
    #         right_arm = False
    #         left_arm = False

    #         if RIGHT_ARM_JOINT_PREFIX in arm.name[0]:
    #             right_arm = True
    #         elif LEFT_ARM_JOINT_PREFIX in arm.name[0]:
    #             left_arm = True
    #         else:
    #             return 
            
    #         q1 = arm.position[0]
    #         if q1>=0:
    #             frontal_elevation_flexion = q1*180/np.pi
    #             frontal_elevation_extension = 0
    #         else:
    #             frontal_elevation_flexion = 0
    #             frontal_elevation_extension = -q1*180/np.pi

    #         abduction = arm.position[1]*180/np.pi #q2
    #         elbow_pronosupination = np.abs(arm.position[2])*180/np.pi #q3
    #         elbow_flexion = np.abs((np.abs(arm.position[3])*180/np.pi-270)) #q4 -> 0° when elbow is bent, olny positive values

    #         position = [frontal_elevation_flexion,frontal_elevation_extension,
    #                               abduction,elbow_pronosupination,elbow_flexion]
            
    #         print("right_arm: ",right_arm)
    #         print("left_arm: ",left_arm)

    #         if right_arm:
    #             self.right_arm.position = position
    #             self.right_arm.header = arm.header
    #         elif left_arm:
    #             self.left_arm.position = position
    #             self.left_arm.header = arm.header
    #         else:
    #             raise Exception("Error in the arm's joints message")

    def computeFrontal(self,tf_elbow_in_shoulder):
        #Get the vector from shoulder to elbow
        v = np.array([tf_elbow_in_shoulder.transform.translation.x,
                      tf_elbow_in_shoulder.transform.translation.y,
                      tf_elbow_in_shoulder.transform.translation.z])

        #Project the vector on the y-z plane
        x = np.array([1,0,0])
        v_prj = self.project_onto_plane(v,x)

        #Get the angle between the vector projected and the y axis
        y = -np.array([0,1,0]) #towards the floor
        angle = self.angle_between_vectors(v_prj,y)

        if(v_prj[2]<0):
            angle = -angle

        return angle*180/np.pi
    
    def computeLateral(self,tf_elbow_in_shoulder):
        #Get the vector from shoulder to elbow
        v = np.array([tf_elbow_in_shoulder.transform.translation.x,
                      tf_elbow_in_shoulder.transform.translation.y,
                      tf_elbow_in_shoulder.transform.translation.z])

        #Project the vector on the y-z plane
        z = np.array([0,0,1])
        v_prj = self.project_onto_plane(v,z)

        #Get the angle between the vector projected and the y axis
        y = -np.array([0,1,0]) #towards the floor
        angle = self.angle_between_vectors(v_prj,y)

        return angle*180/np.pi
    
    def computeFlexion(self,tf_wrist_in_elbow):
        #Get the vector from shoulder to elbow
        v = np.array([tf_wrist_in_elbow.transform.translation.x,
                      tf_wrist_in_elbow.transform.translation.y,
                      tf_wrist_in_elbow.transform.translation.z])

        #Project the vector on the y-z plane
        z = np.array([0,0,1])
        v_prj = self.project_onto_plane(v,z)

        #Get the angle between the vector projected and the y axis
        y = -np.array([0,1,0]) #towards the floor
        angle = self.angle_between_vectors(v_prj,y)

        return angle*180/np.pi
    
    def computeRotation(self,tf_wrist_in_shoulder):
        #Get the vector from shoulder to elbow
        quat = tf_wrist_in_shoulder.transform.rotation
        matrix = R.from_quat([quat.x,quat.y,quat.z,quat.w]).as_matrix()
        v = matrix[:,2] #wrist z axis emerging from the hand's torso

        #Get the angle between the vector projected and the y axis
        z = -np.array([0,0,1])
        angle = self.angle_between_vectors(v,z)

        return angle*180/np.pi

if __name__ == "__main__":
    rospy.init_node("ocra_angles", anonymous=True)
    ocra_angles = OcraAngles()

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rate.sleep()
        ocra_angles.computeAngles()
