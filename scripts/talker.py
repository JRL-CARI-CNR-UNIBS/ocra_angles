#! /usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, Transform, Vector3, TransformStamped
import pandas as pd
from datetime import datetime
from os.path import exists, abspath


#USER PARAMETERS
BACK_INCLINATION_ERROR = 15.6 #degrees
WRITE_TO_CSV = True
DELTA_T_TF = 1 # seconds

#Angles names
ARM_JOINTS_NAMES = ["Frontal", "Lateral","Rotation","Flexion"]
LEFT_ARM_JOINT_PREFIX = "left_arm"
RIGHT_ARM_JOINT_PREFIX = "right_arm"

class OcraAngles():
    def __init__(self):
        self.right_arm_flag = rospy.get_param('~right_arm', True)
        self.left_arm_flag = rospy.get_param('~left_arm', True)
        self.torso_flag = rospy.get_param('~torso_arm', True)

        self.right_arm = JointState()
        self.right_arm.name = ARM_JOINTS_NAMES
        self.right_arm.header.stamp = rospy.Time.now()
        self.right_arm.position = None
        self.right_arm.velocity = [0.0,0.0,0.0,0.0] #does not matter
        self.right_arm.effort   = [0.0,0.0,0.0,0.0] #does not matter

        self.left_arm = JointState()
        self.left_arm.name = ARM_JOINTS_NAMES
        self.left_arm.header.stamp = rospy.Time.now()
        self.left_arm.position = None
        self.left_arm.velocity = [0.0,0.0,0.0,0.0] #does not matter
        self.left_arm.effort   = [0.0,0.0,0.0,0.0] #does not matter

        self.torso = None
    
        if(self.right_arm_flag):
            self.right_arm_pub = rospy.Publisher(RIGHT_ARM_TOPIC, JointState, queue_size=10)
        if(self.left_arm_flag):
            self.left_arm_pub = rospy.Publisher(LEFT_ARM_TOPIC, JointState, queue_size=10)
        if(self.torso_flag):
            self.torso_pub = rospy.Publisher(TORSO_TOPIC, Float64, queue_size=10)

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

    def computeAngles(self):

        if(self.left_arm_flag):
            try:
                tf_left_elbow = self.tfBuffer.lookup_transform(LEFT_SHOULDER_TF_NAME,LEFT_ELBOW_TF_NAME,rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_left_elbow = None

        if not self.left_arm_flag or (tf_left_elbow != None and (rospy.Time().now().secs-tf_left_elbow.header.stamp.secs)>DELTA_T_TF):
            tf_left_elbow = None
        
        if(self.right_arm_flag):
            try:
                tf_right_elbow = self.tfBuffer.lookup_transform(RIGHT_SHOULDER_TF_NAME,RIGHT_ELBOW_TF_NAME,rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_right_elbow = None

        if not self.right_arm_flag or (tf_right_elbow != None and (rospy.Time().now().secs-tf_right_elbow.header.stamp.secs)>DELTA_T_TF):
            tf_right_elbow = None

        if(self.left_arm_flag):
            try:
                tf_left_flexion = self.tfBuffer.lookup_transform(LEFT_ELBOW_TF_NAME,LEFT_WRIST_TF_NAME,rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_left_flexion = None
        
        if not self.left_arm_flag or (tf_left_flexion != None and (rospy.Time().now().secs-tf_left_flexion.header.stamp.secs)>DELTA_T_TF):
            tf_left_flexion = None

        if(self.right_arm_flag):
            try:
                tf_right_flexion = self.tfBuffer.lookup_transform(RIGHT_ELBOW_TF_NAME,RIGHT_WRIST_TF_NAME,rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_right_flexion = None

        if not self.right_arm_flag or (tf_right_flexion != None and (rospy.Time().now().secs-tf_right_flexion.header.stamp.secs)>DELTA_T_TF):
            tf_right_flexion = None

        if(self.left_arm_flag):
            try:
                tf_left_rotation = self.tfBuffer.lookup_transform(LEFT_ELBOW_TF_NAME,LEFT_WRIST_TF_NAME,rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_left_rotation = None

        if not self.left_arm_flag or (tf_left_rotation != None and (rospy.Time().now().secs-tf_left_rotation.header.stamp.secs)>DELTA_T_TF):
            tf_left_rotation = None

        if(self.right_arm_flag):
            try:
                tf_right_rotation = self.tfBuffer.lookup_transform(RIGHT_ELBOW_TF_NAME,RIGHT_WRIST_TF_NAME,rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_right_rotation = None

        if not self.right_arm_flag or (tf_right_rotation != None and (rospy.Time().now().secs-tf_right_rotation.header.stamp.secs)>DELTA_T_TF):
            tf_right_rotation = None
        
        if(self.torso_flag):
            try:
                tf_hip = self.tfBuffer.lookup_transform('world',HIP_TF_NAME,rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_hip = None

        if not self.torso_flag or (tf_hip != None and (rospy.Time().now().secs-tf_hip.header.stamp.secs)>DELTA_T_TF):
            tf_hip = None

        # Compute frame torso_world, with z as tf_torso but y as world
        if tf_hip != None:
            t0     = Transform()
            t0TS   = TransformStamped()

            quat   = tf_hip.transform.rotation
            tran   = [tf_hip.transform.translation.x,tf_hip.transform.translation.y,tf_hip.transform.translation.z]

            matrix = R.from_quat([quat.x,quat.y,quat.z,quat.w]).as_matrix()
            z_axis = matrix[:,2]
            z_axis = self.unit_vector(z_axis)
            y_axis = np.array([0,0,1])
            x_axis = np.cross(y_axis,z_axis)

            mat_r  = np.array([x_axis,y_axis,z_axis]).T
            r      = R.from_matrix(mat_r)
            quat_r = r.as_quat()

            t0.rotation=Quaternion(quat_r[0],quat_r[1],quat_r[2],quat_r[3])
            t0.translation=Vector3(tran[0],tran[1],tran[2])
            t0TS.header.frame_id = "world"
            t0TS.header.stamp= rospy.Time.now()
            t0TS.child_frame_id = CAMERA+"/hip_as_world"
            t0TS.transform=t0
            br = tf2_ros.TransformBroadcaster()
            br.sendTransform(t0TS)

            try:
                tf_neck_to_hip_as_world = self.tfBuffer.lookup_transform(t0TS.child_frame_id,NECK_TF_NAME,rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_neck_to_hip_as_world = None
        else:
            tf_neck_to_hip_as_world = None

        # Left arm
        if tf_left_elbow != None and tf_left_flexion != None and tf_left_rotation != None:
            left_frontal = self.computeFrontal(tf_left_elbow) + BACK_INCLINATION_ERROR
            left_lateral = self.computeLateral(tf_left_elbow)
            left_rotation = np.abs(self.computeRotation(tf_left_rotation)-180)
            left_flexion = self.computeFlexion(tf_left_flexion)
            left_arm = [left_frontal,left_lateral,left_rotation,left_flexion]
            self.left_arm.position = left_arm
        else:
            self.left_arm.position = None

        # Right arm
        if tf_right_elbow != None and tf_right_flexion != None and tf_right_rotation != None:
            right_frontal = -(self.computeFrontal(tf_right_elbow) - BACK_INCLINATION_ERROR)
            right_lateral = self.computeLateral(tf_right_elbow)
            right_rotation = np.abs(self.computeRotation(tf_right_rotation))
            right_flexion = self.computeFlexion(tf_right_flexion)
            right_arm = [right_frontal,right_lateral,right_rotation,right_flexion]
            self.right_arm.position = right_arm
        else:
            self.right_arm.position = None
        
        # Torso
        if tf_neck_to_hip_as_world != None:
            self.torso = self.computeTorso(tf_neck_to_hip_as_world) + BACK_INCLINATION_ERROR
        else:
            self.torso = None

        # Publish
        if self.left_arm.position != None:
            self.left_arm_pub.publish(self.left_arm)
            if WRITE_TO_CSV:
                data = {"Left_"+ARM_JOINTS_NAMES[0]: [self.left_arm.position[0]], "Left_"+ARM_JOINTS_NAMES[1]: [self.left_arm.position[1]],
                         "Left_"+ARM_JOINTS_NAMES[2]: [self.left_arm.position[2]], "Left_"+ARM_JOINTS_NAMES[3]: [self.left_arm.position[3]]}
                df = pd.DataFrame(data) 
                df.insert(0, "Time", datetime.now())
                left_arm_csv_header = not exists(LEFT_ARM_FILE_NAME)
                df.to_csv(LEFT_ARM_FILE_NAME, mode='a', index=False, header = left_arm_csv_header)

        if self.right_arm.position != None:
            self.right_arm_pub.publish(self.right_arm)
            if WRITE_TO_CSV:
                data = {"Right_"+ARM_JOINTS_NAMES[0]: [self.right_arm.position[0]], "Right_"+ARM_JOINTS_NAMES[1]: [self.right_arm.position[1]],
                         "Right_"+ARM_JOINTS_NAMES[2]: [self.right_arm.position[2]], "Right_"+ARM_JOINTS_NAMES[3]: [self.right_arm.position[3]]}
                df = pd.DataFrame(data) 
                df.insert(0, "Time", datetime.now())
                right_arm_csv_header = not exists(RIGHT_ARM_FILE_NAME)
                df.to_csv(RIGHT_ARM_FILE_NAME, mode='a', index=False, header = right_arm_csv_header)
        
        if self.torso != None:
            self.torso_pub.publish(self.torso)
            if WRITE_TO_CSV:
                data = {'Torso': [self.torso]}
                df = pd.DataFrame(data) 
                df.insert(0, "Time", datetime.now())
                torso_csv_header = not exists(TORSO_FILE_NAME)
                df.to_csv(TORSO_FILE_NAME, mode='a', index=False, header = torso_csv_header)

    def computeTorso(self, tf_torso):
        #Get the vector from hip to neck
        v = np.array([tf_torso.transform.translation.x,
                      tf_torso.transform.translation.y,
                      tf_torso.transform.translation.z])
        
        #Project the vector on the y-x plane
        z = np.array([0,0,1])
        v_prj = self.project_onto_plane(v,z)

        #Get the angle between the vector projected and the y axis
        y = np.array([0,1,0])  #towards the floor

        angle= self.angle_between_vectors(v_prj,y)*180/np.pi

        if(v[0]<0):
            angle = -angle

        return angle
        
    def computeFrontal(self,tf_elbow_in_shoulder):
        #Get the vector from shoulder to elbow
        v = np.array([tf_elbow_in_shoulder.transform.translation.x,
                      tf_elbow_in_shoulder.transform.translation.y,
                      tf_elbow_in_shoulder.transform.translation.z])

        #Project the vector on the y-x plane
        z = np.array([0,0,1])
        v_prj = self.project_onto_plane(v,z)

        #Get the angle between the vector projected and the y axis
        y = -np.array([0,1,0])  #towards the floor

        angle = self.angle_between_vectors(v_prj,y) *180/np.pi

        if(v_prj[0]<0):
            angle = -angle

        return angle
    
    def computeLateral(self,tf_elbow_in_shoulder):
        #Get the vector from shoulder to elbow
        v = np.array([tf_elbow_in_shoulder.transform.translation.x,
                      tf_elbow_in_shoulder.transform.translation.y,
                      tf_elbow_in_shoulder.transform.translation.z])

        #Project the vector on the y-z plane
        x = np.array([1,0,0])
        v_prj = self.project_onto_plane(v,x)

        #Get the angle between the vector projected and the y axis
        y = -np.array([0,1,0]) #towards the floor
        angle = self.angle_between_vectors(v_prj,y) *180/np.pi

        return angle
    
    def computeFlexion(self,tf_wrist_in_elbow):
        #Get y axis on wrist with respect to elbow
        quat = tf_wrist_in_elbow.transform.rotation
        matrix = R.from_quat([quat.x,quat.y,quat.z,quat.w]).as_matrix()
        y = matrix[:,1]

        y = np.array(y) #wrist y axis
        y_origin = np.array([0,1,0]) #elbow y axis

        #Get the angle between the y axis of elbow and the y axis of wrist
        return np.abs(self.angle_between_vectors(y,y_origin)*180/np.pi -90)
    
    def computeRotation(self,tf_wrist_in_elbow):
        #Get the vector from shoulder to elbow
        quat = tf_wrist_in_elbow.transform.rotation
        matrix = R.from_quat([quat.x,quat.y,quat.z,quat.w]).as_matrix()
        v = matrix[:,2] #wrist z axis emerging from the hand's torso

        #Get the angle between the vector projected and the y axis
        z = np.array([0,0,1])
        angle = self.angle_between_vectors(v,z)*180/np.pi

        return angle
    
    def unit_vector(self,v):
        return (v / np.linalg.norm(v))

    def project_onto_plane(self, v1, n):
        # Subtract from vector v1 its component along n, the normal vector to the plane
        unit_n = self.unit_vector(n)      
        prj_n = np.dot(v1, unit_n)*unit_n
        res = v1-prj_n

        return res
    
    def angle_between_vectors(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u),-1.0, 1.0))

if __name__ == "__main__":
    rospy.init_node("ocra_angles", anonymous=True)

    #Camera namespace
    global CAMERA
    CAMERA = rospy.get_param("~camera_ns", "")

    print("Camera namespace: ", CAMERA)

    DATE_STR = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    global LEFT_ARM_FILE_NAME, RIGHT_ARM_FILE_NAME, TORSO_FILE_NAME
    PATH_TO_FILE = "/home/galois/projects/sharework_ws/src/ocra_angles/data/"
    LEFT_ARM_FILE_NAME = PATH_TO_FILE+DATE_STR+CAMERA+"_ocra_left_arm.csv"
    RIGHT_ARM_FILE_NAME = PATH_TO_FILE+DATE_STR+CAMERA+"_ocra_right_arm.csv"
    TORSO_FILE_NAME = PATH_TO_FILE+DATE_STR+CAMERA+"_ocra_torso.csv"

    #Publish on
    global RIGHT_ARM_TOPIC, LEFT_ARM_TOPIC, TORSO_TOPIC
    RIGHT_ARM_TOPIC = "ocra/"+CAMERA+"/right_arm_angles"
    LEFT_ARM_TOPIC = "ocra/"+CAMERA+"/left_arm_angles"
    TORSO_TOPIC = "ocra/"+CAMERA+"/torso"

    #Tf
    global NECK_TF_NAME, HIP_TF_NAME, LEFT_SHOULDER_TF_NAME, RIGHT_SHOULDER_TF_NAME, LEFT_ELBOW_TF_NAME, RIGHT_ELBOW_TF_NAME, LEFT_WRIST_TF_NAME, RIGHT_WRIST_TF_NAME
    NECK_TF_NAME = CAMERA+"/neck"
    HIP_TF_NAME = CAMERA+"/hip"
    LEFT_SHOULDER_TF_NAME = CAMERA+"/left_shoulder"
    RIGHT_SHOULDER_TF_NAME = CAMERA+"/right_shoulder"
    LEFT_ELBOW_TF_NAME = CAMERA+"/left_shoulder_elbow"
    RIGHT_ELBOW_TF_NAME = CAMERA+"/right_shoulder_elbow"
    LEFT_WRIST_TF_NAME = CAMERA+"/left_shoulder_wrist"
    RIGHT_WRIST_TF_NAME = CAMERA+"/right_shoulder_wrist"
    
    ocra_angles = OcraAngles()

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rate.sleep()
        ocra_angles.computeAngles()
