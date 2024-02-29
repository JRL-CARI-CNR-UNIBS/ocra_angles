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
from visualization_msgs.msg import Marker, MarkerArray

#USER PARAMETERS
BACK_INCLINATION_ERROR = 15.6 #degrees
WRITE_TO_CSV = True
DELTA_T_TF = 1 # seconds

#Angles names
RIGHT_HAND_ID = 20 #from mediapipe
LEFT_HAND_ID = 19 #from mediapipe
ARM_JOINTS_NAMES = ["Frontal", "Lateral","Rotation","Flexion"]
HAND_JOINTS_NAMES = ["Up/Down", "Left/Right"]
LEFT_ARM_JOINT_PREFIX = "left_arm"
RIGHT_ARM_JOINT_PREFIX = "right_arm"

class OcraAngles():
    def __init__(self):
        self.right_arm_flag = rospy.get_param('~right_arm', True)
        self.left_arm_flag = rospy.get_param('~left_arm', True)
        self.torso_flag = rospy.get_param('~torso_arm', True)
        self.right_hand_flag = rospy.get_param('~right_hand', True)
        self.left_hand_flag = rospy.get_param('~left_hand', True)

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

        self.right_hand_pose = None
        self.left_hand_pose = None
        self.right_hand_time = 0
        self.left_hand_time = 0

        self.right_hand = JointState()
        self.right_hand.name = HAND_JOINTS_NAMES
        self.right_hand.header.stamp = rospy.Time.now()
        self.right_hand.position = None
        self.right_hand.velocity = [0.0,0.0] #does not matter
        self.right_hand.effort   = [0.0,0.0] #does not matter

        self.left_hand = JointState()
        self.left_hand.name = HAND_JOINTS_NAMES
        self.left_hand.header.stamp = rospy.Time.now()
        self.left_hand.position = None
        self.left_hand.velocity = [0.0,0.0] #does not matter
        self.left_hand.effort   = [0.0,0.0] #does not matter

        if(self.right_arm_flag):
            self.right_arm_pub = rospy.Publisher(RIGHT_ARM_TOPIC, JointState, queue_size=10)
        if(self.left_arm_flag):
            self.left_arm_pub = rospy.Publisher(LEFT_ARM_TOPIC, JointState, queue_size=10)
        if(self.torso_flag):
            self.torso_pub = rospy.Publisher(TORSO_TOPIC, Float64, queue_size=10)
        if(self.left_hand_flag):
            self.left_hand_pub = rospy.Publisher(LEFT_HAND_TOPIC, JointState, queue_size=10)
        if(self.right_hand_flag):
            self.right_hand_pub = rospy.Publisher(RIGHT_HAND_TOPIC, JointState, queue_size=10)
        if(self.right_hand_flag or self.left_hand_flag):
            self.keypoints_listener = rospy.Subscriber(CAMERA+"/skeleton_marker", MarkerArray, self.callback_listener)

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    def callback_listener(self,data):
        markers = data.markers
        for marker in markers:
            if(marker.id == RIGHT_HAND_ID):
                self.right_hand_pose = marker.pose
                self.right_hand_time = rospy.Time.now().secs
            elif(marker.id == LEFT_HAND_ID):
                self.left_hand_pose = marker.pose
                self.left_hand_time = rospy.Time.now().secs
        
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
            self.tf_broadcaster.sendTransform(t0TS)

            try:
                tf_neck_to_hip_as_world = self.tfBuffer.lookup_transform(t0TS.child_frame_id,NECK_TF_NAME,rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_neck_to_hip_as_world = None
        else:
            tf_neck_to_hip_as_world = None
        
        if(rospy.Time().now().secs-self.left_hand_time<=DELTA_T_TF):
            t1     = Transform()
            t1TS   = TransformStamped()

            quat1   = self.left_hand_pose.orientation
            tran1   = self.left_hand_pose.position

            t1.rotation=quat1
            t1.translation=tran1
            t1TS.header.frame_id = CAMERA_TF_NAME
            t1TS.header.stamp= rospy.Time.now()
            t1TS.child_frame_id = CAMERA+"/left_hand"
            t1TS.transform=t1
            self.tf_broadcaster.sendTransform(t1TS)
            
            try:
                tf_left_hand = self.tfBuffer.lookup_transform(t1TS.child_frame_id,LEFT_WRIST_TF_NAME,rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_left_hand = None
        else:
            tf_left_hand = None

        if(rospy.Time().now().secs-self.right_hand_time<=DELTA_T_TF):
            t2     = Transform()
            t2TS   = TransformStamped()

            quat2   = self.right_hand_pose.orientation
            tran2   = self.right_hand_pose.position

            t2.rotation=quat2
            t2.translation=tran2
            t2TS.header.frame_id = CAMERA_TF_NAME
            t2TS.header.stamp= rospy.Time.now()
            t2TS.child_frame_id = CAMERA+"/right_hand"
            t2TS.transform=t2
            self.tf_broadcaster.sendTransform(t2TS)
            
            try:
                tf_right_hand = self.tfBuffer.lookup_transform(t2TS.child_frame_id,RIGHT_WRIST_TF_NAME,rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                tf_right_hand = None
        else:
            tf_right_hand = None

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
        
        # Left Hand
        if tf_left_hand != None:
            self.left_hand.position = self.computeHand(tf_left_hand)
            
        else:
            self.left_hand.position = None
        
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
        
        # Right hand
        if tf_right_hand != None:
            self.right_hand.position = self.computeHand(tf_right_hand)
        else:
            self.right_hand.position = None
        
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

        if self.right_hand.position != None:
            self.right_hand_pub.publish(self.right_hand)
            if WRITE_TO_CSV:
                data = {"Right_"+HAND_JOINTS_NAMES[0]: [self.right_hand.position[0]], "Right_"+HAND_JOINTS_NAMES[1]: [self.right_hand.position[1]]}
                df = pd.DataFrame(data) 
                df.insert(0, "Time", datetime.now())
                right_hand_csv_header = not exists(RIGHT_HAND_FILE_NAME)
                df.to_csv(RIGHT_HAND_FILE_NAME, mode='a', index=False, header = right_hand_csv_header)

        if self.left_hand.position != None:
            self.left_hand_pub.publish(self.left_hand)
            if WRITE_TO_CSV:
                data = {"Left_"+HAND_JOINTS_NAMES[0]: [self.left_hand.position[0]], "Left_"+HAND_JOINTS_NAMES[1]: [self.left_hand.position[1]]}
                df = pd.DataFrame(data) 
                df.insert(0, "Time", datetime.now())
                left_hand_csv_header = not exists(LEFT_HAND_FILE_NAME)
                df.to_csv(LEFT_HAND_FILE_NAME, mode='a', index=False, header = left_hand_csv_header)
        
        if self.torso != None:
            self.torso_pub.publish(self.torso)
            if WRITE_TO_CSV:
                data = {'Torso': [self.torso]}
                df = pd.DataFrame(data) 
                df.insert(0, "Time", datetime.now())
                torso_csv_header = not exists(TORSO_FILE_NAME)
                df.to_csv(TORSO_FILE_NAME, mode='a', index=False, header = torso_csv_header)

        # self.left_hand_available = False
        # self.right_hand_available = False

    def computeHand(self, tf_hand):
        #Get the vector from wrist to hand
        v = np.array([tf_hand.transform.translation.x,
                      tf_hand.transform.translation.y,
                      tf_hand.transform.translation.z])

        #Project the vector on the y-z plane
        x = np.array([1,0,0])
        v_prj = self.project_onto_plane(v,x)

        #Get the angle between the vector projected and the y axis
        y = -np.array([0,1,0])  #towards the floor

        angle_up_down = np.abs(self.angle_between_vectors(v_prj,y) *180/np.pi)

        #Project the vector on the y-x plane
        z = np.array([0,0,1])
        v_prj2 = self.project_onto_plane(v,z)

        angle_left_right = np.abs(self.angle_between_vectors(v_prj2,y) *180/np.pi)

        return [angle_up_down,angle_left_right]

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
    
    global LEFT_ARM_FILE_NAME, RIGHT_ARM_FILE_NAME, TORSO_FILE_NAME, LEFT_HND_FILE_NAME, RIGHT_HAND_FILE_NAME
    PATH_TO_FILE = "/home/galois/projects/sharework_ws/src/ocra_angles/data/"
    LEFT_ARM_FILE_NAME = PATH_TO_FILE+DATE_STR+CAMERA+"_ocra_left_arm.csv"
    RIGHT_ARM_FILE_NAME = PATH_TO_FILE+DATE_STR+CAMERA+"_ocra_right_arm.csv"
    TORSO_FILE_NAME = PATH_TO_FILE+DATE_STR+CAMERA+"_ocra_torso.csv"
    LEFT_HAND_FILE_NAME = LEFT_ARM_FILE_NAME = PATH_TO_FILE+DATE_STR+CAMERA+"_ocra_left_hand.csv"
    RIGHT_HAND_FILE_NAME = PATH_TO_FILE+DATE_STR+CAMERA+"_ocra_right_hand.csv"

    #Publish on
    global RIGHT_ARM_TOPIC, LEFT_ARM_TOPIC, TORSO_TOPIC, LEFT_HAND_TOPIC, RIGHT_HAND_TOPIC
    RIGHT_ARM_TOPIC = "ocra/"+CAMERA+"/right_arm_angles"
    LEFT_ARM_TOPIC = "ocra/"+CAMERA+"/left_arm_angles"
    RIGHT_HAND_TOPIC = "ocra/"+CAMERA+"/right_hand_angles"
    LEFT_HAND_TOPIC = "ocra/"+CAMERA+"/left_hand_angles"
    TORSO_TOPIC = "ocra/"+CAMERA+"/torso"

    #Tf
    global CAMERA_TF_NAME, NECK_TF_NAME, HIP_TF_NAME, LEFT_SHOULDER_TF_NAME, RIGHT_SHOULDER_TF_NAME, LEFT_ELBOW_TF_NAME, RIGHT_ELBOW_TF_NAME, LEFT_WRIST_TF_NAME, RIGHT_WRIST_TF_NAME
    NECK_TF_NAME = CAMERA+"/neck"
    HIP_TF_NAME = CAMERA+"/hip"
    LEFT_SHOULDER_TF_NAME = CAMERA+"/left_shoulder"
    RIGHT_SHOULDER_TF_NAME = CAMERA+"/right_shoulder"
    LEFT_ELBOW_TF_NAME = CAMERA+"/left_shoulder_elbow"
    RIGHT_ELBOW_TF_NAME = CAMERA+"/right_shoulder_elbow"
    LEFT_WRIST_TF_NAME = CAMERA+"/left_shoulder_wrist"
    RIGHT_WRIST_TF_NAME = CAMERA+"/right_shoulder_wrist"
    CAMERA_TF_NAME = CAMERA+"_color_optical_frame"
    
    ocra_angles = OcraAngles()

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rate.sleep()
        ocra_angles.computeAngles()
