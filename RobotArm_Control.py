#!/usr/bin/env python

from copy import deepcopy
import math
import numpy
import random
from threading import Thread, Lock
import sys
import matplotlib.pyplot as plt
from datetime import datetime #Added this line to measure execution time
import actionlib
import control_msgs.msg
import geometry_msgs.msg
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
import moveit_commander
import moveit_msgs.msg
import moveit_msgs.srv
import rospy
import sensor_msgs.msg
import tf
import trajectory_msgs.msg
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker
import time


def convert_to_message(T):
    t = geometry_msgs.msg.Pose()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[0]
    t.orientation.y = orientation[1]
    t.orientation.z = orientation[2]
    t.orientation.w = orientation[3]        
    return t

def convert_from_message(msg):
    R = tf.transformations.quaternion_matrix((msg.orientation.x,
                                              msg.orientation.y,
                                              msg.orientation.z,
                                              msg.orientation.w))
    T = tf.transformations.translation_matrix((msg.position.x, 
                                               msg.position.y, 
                                               msg.position.z))
    return numpy.dot(T,R)

class RRTNode(object):
    def __init__(self):
        self.q=numpy.zeros(7)
        self.parent = None

class MoveArm(object):

    def __init__(self):
        print "HW3 initializing..."
        # Prepare the mutex for synchronization
        self.mutex = Lock()

        # min and max joint values are not read in Python urdf, so we must hard-code them here
        self.q_min = []
        self.q_max = []
        self.q_min.append(-1.700);self.q_max.append(1.700)
        self.q_min.append(-2.147);self.q_max.append(1.047)
        self.q_min.append(-3.054);self.q_max.append(3.054)
        self.q_min.append(-0.050);self.q_max.append(2.618)
        self.q_min.append(-3.059);self.q_max.append(3.059)
        self.q_min.append(-1.570);self.q_max.append(2.094)
        self.q_min.append(-3.059);self.q_max.append(3.059)

        # Subscribes to information about what the current joint values are.
        rospy.Subscriber("robot/joint_states", sensor_msgs.msg.JointState, self.joint_states_callback)

        # Initialize variables
        self.q_current = []
        self.joint_state = sensor_msgs.msg.JointState()

        # Create interactive marker
        self.init_marker()

        # Connect to trajectory execution action
        self.trajectory_client = actionlib.SimpleActionClient('/robot/limb/left/follow_joint_trajectory', 
                                                              control_msgs.msg.FollowJointTrajectoryAction)
        self.trajectory_client.wait_for_server()
        print "Joint trajectory client connected"

        # Wait for moveit IK service
        rospy.wait_for_service("compute_ik")
        self.ik_service = rospy.ServiceProxy('compute_ik',  moveit_msgs.srv.GetPositionIK)
        print "IK service ready"

        # Wait for validity check service
        rospy.wait_for_service("check_state_validity")
        self.state_valid_service = rospy.ServiceProxy('check_state_validity',  
                                                      moveit_msgs.srv.GetStateValidity)
        print "State validity service ready"

        # Initialize MoveIt
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm") 
        print "MoveIt! interface ready"

        # How finely to sample each joint
        self.q_sample = [0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 0.4]
        self.joint_names = ["left_s0", "left_s1",
                            "left_e0", "left_e1",
                            "left_w0", "left_w1","left_w2"]

        # Options
        self.subsample_trajectory = True
        self.spline_timing = True
        self.show_plots = False

        print "Initialization done."


    def control_marker_feedback(self, feedback):
        pass

    def get_joint_val(self, joint_state, name):
        if name not in joint_state.name:
            print "ERROR: joint name not found"
            return 0
        i = joint_state.name.index(name)
        return joint_state.position[i]

    def set_joint_val(self, joint_state, q, name):
        if name not in joint_state.name:
            print "ERROR: joint name not found"
        i = joint_state.name.index(name)
        joint_state.position[i] = q

    """ Given a complete joint_state data structure, this function finds the values for 
    a particular set of joints in a particular order (in our case, the left arm joints ordered
    from proximal to distal) and returns a list q[] containing just those values.
    """
    def q_from_joint_state(self, joint_state):
        q = []
        q.append(self.get_joint_val(joint_state, "left_s0"))
        q.append(self.get_joint_val(joint_state, "left_s1"))
        q.append(self.get_joint_val(joint_state, "left_e0"))
        q.append(self.get_joint_val(joint_state, "left_e1"))
        q.append(self.get_joint_val(joint_state, "left_w0"))
        q.append(self.get_joint_val(joint_state, "left_w1"))
        q.append(self.get_joint_val(joint_state, "left_w2"))
        return q

    """ Given a list q[] of joint values and an already populated joint_state, this function assumes 
    that the passed in values are for a particular set of joints in a particular order (in our case,
    the left arm joints ordered from proximal to distal) and edits the joint_state data structure to
    set the values to the ones passed in.
    """
    def joint_state_from_q(self, joint_state, q):
        self.set_joint_val(joint_state, q[0], "left_s0")
        self.set_joint_val(joint_state, q[1], "left_s1")
        self.set_joint_val(joint_state, q[2], "left_e0")
        self.set_joint_val(joint_state, q[3], "left_e1")
        self.set_joint_val(joint_state, q[4], "left_w0")
        self.set_joint_val(joint_state, q[5], "left_w1")
        self.set_joint_val(joint_state, q[6], "left_w2")        

    """ Creates simple timing information for a trajectory, where each point has velocity
    and acceleration 0 for all joints, and all segments take the same amount of time
    to execute.
    """
    def compute_simple_timing(self, q_list, time_per_segment):
        v_list = [numpy.zeros(7) for i in range(0,len(q_list))]
        a_list = [numpy.zeros(7) for i in range(0,len(q_list))]
        t = [i*time_per_segment for i in range(0,len(q_list))]
        return v_list, a_list, t

    """ This function will perform IK for a given transform T of the end-effector. It returs a list q[]
    of 7 values, which are the result positions for the 7 joints of the left arm, ordered from proximal
    to distal. If no IK solution is found, it returns an empy list.
    """
    def IK(self, T_goal):
        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request.group_name = "left_arm"
        req.ik_request.robot_state = moveit_msgs.msg.RobotState()
        req.ik_request.robot_state.joint_state = self.joint_state
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped = geometry_msgs.msg.PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = "base"
        req.ik_request.pose_stamped.header.stamp = rospy.get_rostime()
        req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
        req.ik_request.timeout = rospy.Duration(3.0)
        res = self.ik_service(req)
        q = []
        if res.error_code.val == res.error_code.SUCCESS:
            q = self.q_from_joint_state(res.solution.joint_state)
        return q

    """ This function checks if a set of joint angles q[] creates a valid state, or one that is free
    of collisions. The values in q[] are assumed to be values for the joints of the left arm, ordered
    from proximal to distal. 
    """
    def is_state_valid(self, q):
        req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = "left_arm"
        current_joint_state = deepcopy(self.joint_state)
        current_joint_state.position = list(current_joint_state.position)
        self.joint_state_from_q(current_joint_state, q)
        req.robot_state = moveit_msgs.msg.RobotState()
        req.robot_state.joint_state = current_joint_state
        res = self.state_valid_service(req)
        return res.valid

    # This function will plot the position, velocity and acceleration of a joint
    # based on the polynomial coefficients of each segment that makes up the 
    # trajectory.
    # Arguments:
    # - num_segments: the number of segments in the trajectory
    # - coefficients: the coefficients of a cubic polynomial for each segment, arranged
    #   as follows [a_1, b_1, c_1, d_1, ..., a_n, b_n, c_n, d_n], where n is the number
    #   of segments
    # - time_per_segment: the time (in seconds) allocated to each segment.
    # This function will display three plots. Execution will continue only after all 
    # plot windows have been closed.
    def plot_trajectory(self, num_segments, coeffs, time_per_segment):
        resolution = 1.0e-2
        assert(num_segments*4 == len(coeffs))
        t_vec = []
        q_vec = []
        a_vec = []
        v_vec = []
        for i in range(0,num_segments):
            t=0
            while t<time_per_segment:
                q,a,v = self.sample_polynomial(coeffs,i,t)
                t_vec.append(t+i*time_per_segment)
                q_vec.append(q)
                a_vec.append(a)
                v_vec.append(v)
                t = t+resolution
        self.plot_series(t_vec,q_vec,"Position")
        self.plot_series(t_vec,v_vec,"Velocity")
        self.plot_series(t_vec,a_vec,"Acceleration")
        plt.show()

    """ This is the main function to be filled in for HW3.
    Parameters:
    - q_start: the start configuration for the arm
    - q_goal: the goal configuration for the arm
    - q_min and q_max: the min and max values for all the joints in the arm.
    All the above parameters are arrays. Each will have 7 elements, one for each joint in the arm.
    These values correspond to the joints of the arm ordered from proximal (closer to the body) to 
    distal (further from the body). 

    The function must return a trajectory as a tuple (q_list,v_list,a_list,t).
    If the trajectory has n points, then q_list, v_list and a_list must all have n entries. Each
    entry must be an array of size 7, specifying the position, velocity and acceleration for each joint.

    For example, the i-th point of the trajectory is defined by:
    - q_list[i]: an array of 7 numbers specifying position for all joints at trajectory point i
    - v_list[i]: an array of 7 numbers specifying velocity for all joints at trajectory point i
    - a_list[i]: an array of 7 numbers specifying acceleration for all joints at trajectory point i
    Note that q_list, v_list and a_list are all lists of arrays. 
    For example, q_list[i][j] will be the position of the j-th joint (0<j<7) at trajectory point i 
    (0 < i < n).

    For example, a trajectory with just 2 points, starting from all joints at position 0 and 
    ending with all joints at position 1, might look like this:

    q_list=[ numpy.array([0, 0, 0, 0, 0, 0, 0]),
             numpy.array([1, 1, 1, 1, 1, 1, 1]) ]
    v_list=[ numpy.array([0, 0, 0, 0, 0, 0, 0]),
             numpy.array([0, 0, 0, 0, 0, 0, 0]) ]
    a_list=[ numpy.array([0, 0, 0, 0, 0, 0, 0]),
             numpy.array([0, 0, 0, 0, 0, 0, 0]) ]
             
    Note that the trajectory should always begin from the current configuration of the robot.
    Hence, the first entry in q_list should always be equal to q_start. 

    In addition, t must be a list with n entries (where n is the number of points in the trajectory).
    For the i-th trajectory point, t[i] must specify when this point should be reached, relative to
    the start of the trajectory. As a result t[0] should always be 0. For the previous example, if we
    want the second point to be reached 10 seconds after starting the trajectory, we can use:

    t=[0,10]

    When you are done computing all of these, return them using

    return q_list,v_list,a_list,t

    In addition, you can use the function self.is_state_valid(q_test) to test if the joint positions 
    in a given array q_test create a valid (collision-free) state. q_test will be expected to 
    contain 7 elements, each representing a joint position, in the same order as q_start and q_goal.
    """ 

                  
    def motion_plan(self, q_start, q_goal, q_min, q_max):
                
        use_RRT=True #Flag to select motion planner
        
        # - - - - time allowed for PRM to find results - - - - -
        time_allowed = 60
        # - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
        if use_RRT==True:
          q_list=self.RRTPlanner(q_start, q_goal, q_min, q_max)        
        else:
          q_list=self.PRMPlanner(q_start, q_goal, q_min, q_max, time_allowed)

        
        
        # 1 to set your total desired time to get from q_start to q_goal
	# otherwise it will use 1 second segments
	sdt = 0 #set_desired_total_time
	# If the the above is 1, set your total desired time
	ttd = 5 #tot_time_desired
	####### SETTABLE PARAMETERS ########
	# Set to the joint number you want to plot
	joint_num = 6 #Joint to be plotted
	#####################################

        '--Activate this line to run code using our trajectory implementation --------------' 
        v_list, a_list, t = self.Trajectory(q_list, sdt, ttd, joint_num)        

        '-----------------------------------------------------------------------------------'

        # A provided convenience function creates the velocity and acceleration data, 
        # assuming 0 velocity and acceleration at each intermediate point, and 10 seconds
        # for each trajectory segment.
        #v_list,a_list,t = self.compute_simple_timing(q_list, 10)
        return q_list, v_list, a_list, t
        # ---------------------------------------------------------------
        
        '--------------RRT Motion planning----------------------------------'

    def RRTPlanner(self,q_start, q_goal, q_min, q_max):  
        q_list = []
        q_tree = RRTNode() #Instantiation of tree_nodes
        q_tree.q=q_start #Storing first node
        tree_nodes=[]
        tree_nodes.append(q_tree)
        #q_diff=numpy.linalg.norm(numpy.subtract(q_goal,q_start)) #First distance between closest node to target
        N=0 #q_start is the first node of the tree
        goal_reached = False
                
        '--------Reaching goal using RRT--------------------------------------------'
        while goal_reached != True : #goal_reached == False: #This loop will iterate until the distance between closest node to target and target is <=0.05 
          dist=[] #Initializing distance vector (between nodes and q_rand)
          segment_invalid=False #Flag used to break loop if a segment is found invalid
          
          q_rand=numpy.random.uniform(low=q_min, high=q_max) #Sample point (uniform using joints limits)
          while not self.is_state_valid(q_rand): #to make sure that q_rand is not within obstacles 
            q_rand=numpy.random.uniform(low=q_min, high=q_max) #Sample point (uniform using joints limits)  	
     
          '----closest neighbor implementation-----------------------------'
          #mind=numpy.linalg.norm(q_rand-q_start) #Initializing min. distance (qrand and qstart (first node))
          for k in range(0, len(tree_nodes)): #This will find the distances from nodes to random sample
            dist=numpy.linalg.norm(q_rand-tree_nodes[k].q) #calculating distance between nodes in tree and random point
            if k==0 or dist < mind:
              mind=dist #New minimum distance
              q_ind=k #Storing index of node in tree with minimum distance
                   
          dir_qrand=q_rand-tree_nodes[q_ind].q  #vector from closest node to q_rand
          
          '-----Segments Evaluation--------------------------------------------------------------'           

          q_unit_qrand=dir_qrand/numpy.linalg.norm(dir_qrand) #unit vector from node in tree towards q_rand
          for i in range(0,5): #Creating segment towards new qrand point and evaluating its validity
            q_new_sample=tree_nodes[q_ind].q+(0.1*(i+1)*q_unit_qrand) #sample spaced 0.1
            if self.is_state_valid(q_new_sample)==False:
              segment_invalid=True
              break #skips next steps of loop and re-starts for a new q_rand
            if i==4: #All spaced samples are good
              q_tree = RRTNode() #Instantiation of tree_nodes
              q_tree.q=q_new_sample
              q_tree.parent=q_ind
              tree_nodes.append(q_tree) #New node is found and added to the tree
              N=N+1 #nodes counter
              
              print 'There are ', N,' nodes in the RRT'
                        
          if segment_invalid==True: continue  #Goes back to while loop starting
          
          '----------Last node reaches goal?-----------------------'
          dir_qgoal=q_goal-tree_nodes[-1].q #Vector from last node in tree to q_goal
          q_diff_goal=numpy.linalg.norm(dir_qgoal)  #Distance from last node to goal
          seg_to_goal=int(math.floor(q_diff_goal/0.1)) #Number of segments from last node in tree to goal of magnitude 0.1
          
          for k in range(0, seg_to_goal): #Evaluates all the segments of length 0.1 from last node to goal
            q_to_goal=tree_nodes[-1].q+(0.1*(k+1)*dir_qgoal/q_diff_goal) #k segment from node to goal
            if self.is_state_valid(q_to_goal)==False:
              break
            if k==seg_to_goal-1:
              goal_reached = True
              q_tree = RRTNode() #Instantiation of tree_nodes
              q_tree.q=q_goal
              q_tree.parent=len(tree_nodes)-1
              tree_nodes.append(q_tree) #Goal is added to the tree
              print 'goal reached' 
          
          #Maximum number of nodes per tree         
          if N > 2000:
            print "No trajectory found, please try again!"
            sys.exit()
             
        '------------Creating path from start to goal----------------'
        q_s_to_g=[]
        i = len(tree_nodes)-1; #index of last node in the tree (q_goal)
        q_s_to_g.append(tree_nodes[i].q)
        while True:
          i = tree_nodes[i].parent; #index of parent node of current evaluated node
          if i == 0: #when i==0 means the start goal has been reached
            q_s_to_g.append(tree_nodes[i].q) #Adds q_start  
            break
          q_s_to_g.append(tree_nodes[i].q)
        q_s_to_g.reverse()   
        print 'raw path has ', len(q_s_to_g),' nodes' 
        
        '------------Shortcutting path------------------------------'
        q_plen=len(q_s_to_g) 
        q_sc=[]
        q_sc.append(q_s_to_g[0]) #Appends q_start to the shortcut trajectory  
        i=0
        sc_node = False
        sc_done = False
        c=0
        while True: 
          for k in range(i, q_plen-1):
            for l in range(q_plen-1,k,-1):
              c=c+1 #Counter to determine when qgoal is found
              q_dir_seg=q_s_to_g[l]-q_s_to_g[k] #Vector between k node and farthest nodes in path
              q_dir_unit=q_dir_seg/numpy.linalg.norm(q_dir_seg) #unit vector from k node to farthest nodes       
              kseg_to_goal=int(math.floor(numpy.linalg.norm(q_dir_seg)/0.1)) #Number of segments of 0.1 lenght from k node to farthest nodes
              for m in range(0, kseg_to_goal): 
                q_n_to_g=q_s_to_g[k]+0.1*(m+1)*q_dir_unit #m segment from node k to farthest node
                if self.is_state_valid(q_n_to_g)==False:
                  break #Breaks from segment evaluation - no longer valid it will jump to next farthest node
                if m==kseg_to_goal-1:
                  q_sc.append(q_s_to_g[l]) #Appends this far node as the next point in the shortcut path
                  sc_node = True
                  if c==1:
                    sc_done=True
              if sc_node == True: 
                break #Breaks from loop finding farthest nodes  
            c=0
            if sc_node==True: #Breaks from loop sitting at next node of shortcut         
              sc_node = False
              break #Breaks from loop finding segments between nodes        
          i=l #Next for loop will start at new node of shorcut path
          if sc_done==True: break 
                           
        print 'shortcut has ',len(q_sc),' points'  
            
        '-----------Re-sampling------------------------------------------------------------------------'      
        q_list.append(q_sc[0]) #Appending first point to sampled list
        #q_seg=[]
        for i in range(0, len(q_sc)-1):
          q_seg=q_sc[i+1]-q_sc[i] #Vector between k node and farthest nodes in path
          q_dir=q_seg/numpy.linalg.norm(q_seg) #unit vector from k node to farthest nodes       
          segments=int(math.floor(numpy.linalg.norm(q_seg)/0.5)) #Number of segments of 0.5 lenght 
          #print 'distance to sample ',numpy.linalg.norm(q_seg)
          #print 'number points in segment ', segments
          for k in range(0, segments):
            q_new=q_sc[i]+0.5*(k+1)*q_dir #k sampled point
            if k==segments-1:
              q_list.append(q_sc[i+1]) #Adding point in shortcut path rather than the sampled point
            else:
              q_list.append(q_new) #Adding new sampled point to q_list

        print 'Sampled path has ', len(q_list), ' nodes'
        return q_list
    
        '----------Trajectory Calculation---------------------------------------------------------------------' 
   
    def Trajectory(self, q_list, sdt, ttd, joint_num): 
     
	set_desired_total_time = sdt
	tot_time_desired = ttd
	plot_joint_num = joint_num
	##########################################################
	
	######## Generate time vector and other info #############
	n_joints = len(q_list[0][:])
	#print "number of joints"
	#print n_joints
	n_points = len(q_list)
	print "Calculating velocities and accelerations for ", n_points,' points'
	#print n_points
	n_segments = n_points-1
	#print "number of segments"
	#print n_segments
	if set_desired_total_time == 1:	
		time_step_desired = tot_time_desired/(n_points-1)
		t = numpy.multiply(time_step_desired, range(n_points))	
	else:
		time_step_desired = 1
		t = range(n_points)
	print "time vector"
	print t
	##########################################################
	
	########## Generate coeff matrix #########################
	# Vectors needed
	a = numpy.array([0.0,0.0,0.0,1.0]) # Initial position = q_list
	b = numpy.array([0.0,0.0,1.0,0.0]) # Initial velocity = 0
	c = numpy.array([1.0,1.0,1.0,1.0]) # Constrain seg 1 to pt 2 position
	d = numpy.array([3.0,2.0,1.0,0.0]) # Final velocity = 0
	e = numpy.array([3.0,2.0,1.0,0.0,0.0,0.0,-1.0,0.0]) # Constrain vels of seg 1 and 2
	f = numpy.array([6.0,2.0,0.0,0.0,0.0,-2.0,0.0,0.0]) # Constrain accels of seg 1 and 2
	g = numpy.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]) # Constrain seg 2 to pt 2 position
	h = numpy.array([0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0]) # Constrain seg 2 to pt 3 position
	if n_segments == 1:
		coeff_mat = numpy.vstack((a,b,c,d))
	else:	
		blk1_left = numpy.vstack((a,b,c))
		blk1_right = numpy.zeros((3,(n_segments-1)*4))
		blk1 = numpy.hstack((blk1_left,blk1_right))
		#print "blk1"
		#print blk1
		# Second block
		blk2_left = numpy.vstack((e,f,g,h))
		blk2_right = numpy.zeros((4,(n_segments-2)*4))
		blk2 = numpy.hstack((blk2_left,blk2_right))
		#print "blk2"
		#print blk2
		coeff_mat = numpy.vstack((blk1,blk2))
		#print "coeff mat"
		#print coeff_mat
		# Add remaining blocks
		if n_segments >= 3:
			for i in range(n_segments-2):
				zeros_before = numpy.zeros((4,4*(i+1)))
				#print "zeros before"
				#print zeros_before
				zeros_after = numpy.zeros((4,4*(n_segments-i-3)))
				#print "zeros after"
				#print zeros_after
				block_n = numpy.hstack((zeros_before,blk2_left,zeros_after))
				#print "block n"
				#print block_n
				coeff_mat = numpy.vstack((coeff_mat,block_n))
		# Add final block
		final_block = numpy.hstack((numpy.zeros((4*(n_segments-1))),d))
		coeff_mat = numpy.vstack((coeff_mat,final_block))
	#print "coeff mat"
	#print coeff_mat
	###########################################################
	
	#### Invert and duplicate coeff matrix for 7 joints #######
	ict = numpy.linalg.inv(coeff_mat)
	#print "ict"
	#print ict
	ict_7 = []
	for i in range(7):
		ict_7.append(ict)
	#print "ict_7"
	#print ict_7
	###########################################################
	
	### Generate position vector from q_list and calc coeffs ##
	# First segment
	pv_7 = []
	coeffs_7 = []
	for j in range(n_joints):
		pos_vector = numpy.array([[q_list[0][j]],[0.0],[q_list[1][j]]])
		# Remaining segments
		if n_segments >= 2:
			for i in range(n_segments-1):
				pos_vector = numpy.vstack((pos_vector,numpy.array([[0.],[0.]])))
				#print "second pos vector"
				#print pos_vector
				pos_vector = numpy.vstack((pos_vector,q_list[i+1][j],q_list[i+2][j]))
				#print "third pos vector"
				#print pos_vector
		# Final velocity
		pos_vector = numpy.vstack((pos_vector,[0.]))
		#print "pos vector"
		#print pos_vector
		pv_7.append(pos_vector)
		coeffs = numpy.dot(ict,pos_vector)
		if j == plot_joint_num:
			coeffs_plot = coeffs
		#print "coeff"
		#print coeffs
		coeffs_7.append(coeffs)
	#print "pv_7"
	#print pv_7
	#print "coeffs 7"
	#print coeffs_7
	###########################################################
	
	############# Compute velocities ###########################
	#print "a of joint 1 segment 1"
	#print coeffs_7[0][0]
	#print "b of joint 1 segment 1"
	#print coeffs_7[0][1]
	#print "a of joint 2 segment 1"
	#print coeffs_7[1][0]
	#print "b of joint 2 segment 1"
	#print coeffs_7[1][1]
	
	v_list = []
	for j in range(n_segments):
		pt0_sj_vels = []
		# pt 0 velocity of segment j for all joints
		for i in range(n_joints):
			pt0_sj_vels = numpy.hstack((pt0_sj_vels,coeffs_7[i][j*4+2]))
		#print "all joints velocity pt 0"
		#print pt0_sj_vels
		v_list.append(pt0_sj_vels)
	
	# Final velocity
	final_vels = []
	for x in range(n_joints):
		v_term1 = numpy.multiply(3,coeffs_7[x][4*(n_segments-1)])
		v_term2 = numpy.multiply(2,coeffs_7[x][4*(n_segments-1)+1])
		v_term3 = coeffs_7[x][4*(n_segments-1)+2]
		v_total = v_term1[0]+v_term2[0]+v_term3[0]
		final_vels = numpy.hstack((final_vels,v_total))
	v_list.append(final_vels)
	print 'Velocities vector has ', len(v_list), ' points'
	#print v_list
	##############################################################

	############# Compute accelerations ###########################
	a_list = []
	for j in range(n_segments):
		pt0_sj_accels = []
		# pt 0 accel of segment j for all joints
		for i in range(n_joints):
			pt0_sj_accels = numpy.hstack((pt0_sj_accels,numpy.multiply(2,coeffs_7[i][j*4+1])))
		#print "all joints accels pt 0"
		#print pt0_sj_accels
		a_list.append(pt0_sj_accels)

	# Final acceleration
	final_accels = []
	for x in range(n_joints):
		a_term1 = numpy.multiply(6,coeffs_7[x][4*(n_segments-1)])
		#print a_term1
		a_term2 = numpy.multiply(2,coeffs_7[x][4*(n_segments-1)+1])
		#print a_term2
		a_total = a_term1[0]+a_term2[0]
		final_accels= numpy.hstack((final_accels,a_total))
		#print final_accels
	a_list.append(final_accels)
	print 'Acceleration vector has ', len(a_list), ' points'
	#print a_list
	#############################################################
        
        if self.show_plots == True:
	  print 'Plotting position, velocity and acceleration for joint ', joint_num
          self.plot_trajectory(n_segments, coeffs, time_step_desired)
	
	return v_list, a_list, t

    '--------------PRM Motion planning----------------------------------'

    def PRMPlanner(self,q_start, q_goal, q_min, q_max, time_given):  
      
        q_list = []

        start_time = time.time()
        last_dijkstras = 0 # time of most recent dijkstras test
        PRM_points = [q_start, q_goal]
        connections = [[],[]]  # connections[node#] represents [list of connected nodes]
        distances = [[],[]]  # distances[node#] represents [distances of each connected node]
        dijkstras_results = []
        route_home = []
        q_new = q_goal
        while (time.time() < (start_time + time_given)):
            num_nodes = len(PRM_points)

            # - - - - - - - Establish connections with other points - - - - - - 
            for y in range (0, (num_nodes-1)):
                clear_path = True
                q_distance = numpy.linalg.norm(numpy.subtract(PRM_points[y],q_new))
                q_direction = numpy.subtract(PRM_points[y],q_new)
                num_segments = math.ceil(q_distance/0.5)
                q_direction = [b / num_segments for b in q_direction]
                #print "q_new = ", q_new
                #print "PRM Point ",y," = ",PRM_points[y]
                    #check intermediate points to see if there are obstacles in between
                for z in range (0, int(num_segments-1)):
                    q_seg = [c * (z+1.0) for c in q_direction]  #take another small step
                    int_point = numpy.add(q_new,q_seg)
                    if (not self.is_state_valid(int_point)):
                        clear_path = False
                        break
                if (clear_path):
                    connections[(num_nodes - 1)].append(y) 
                    connections[y].append((num_nodes - 1))
                    distances[(num_nodes - 1)].append(q_distance)
                    distances[y].append(q_distance)
            #print "connection between node",(num_nodes-1),"and",connections[(num_nodes - 1)]
            # - - - - - - - ^^ Establish connections ^^ - - - - - - - - - 

            # - - - - - - - List all nodes connected to start - - - - - -
            connected_nodes = [0]
            for x in connected_nodes:
                for y in connections[x]:
                    if y not in connected_nodes:
                        connected_nodes.append(y)
            #print "connected nodes : ",connected_nodes
            # - - - - - - ^^ List nodes connected to start ^^ - - - - - - 

            # - - - - - - - - - run Dijkstras algorithm - - - - - - - - - 
            if (1 in connected_nodes and (time.time() > (last_dijkstras + 1))):
                print "valid path to goal.  Running Dijkstras"
                current_node = 0
                g_values =[0] # shortest distance from start to node
                way_home = [0] # each nodes first step towards the start (shortest way)
                unvisited_nodes = [0]

                for a in range (1,num_nodes):  #initialize needed lists
                    g_values.append(9999)
                    unvisited_nodes.append(a)
                    way_home.append(1)

                while (not current_node == 1):  #calculate g until q_goal has smallest unvisited
                    unvisited_nodes.remove(current_node)
                    for a in connections[current_node]:  #update g of each neighbor
                        d_to_next_node = distances[current_node][connections[current_node].index(a)]
                        if (g_values[a] > (g_values[current_node] + d_to_next_node)):
                            g_values[a] = g_values[current_node] + d_to_next_node
                            way_home[a] = current_node
                    smallest_g = 9999        
                    for b in unvisited_nodes:  # find unvisited node with smallest g
                        if (g_values[b] < smallest_g):
                            smallest_g = g_values[b]
                            current_node = b

                #print "direction home = ",way_home
                if ((len(dijkstras_results)==0) or (g_values[1] < min(dijkstras_results))):
                    q_list=[]
                    current_node = 1
                    route_home = [1]
                    print "faster route found"
                    while (current_node != 0):  # follow path back until arriving at q_start
                        q_list.insert(0,PRM_points[current_node]) #add point to front of list
                        current_node = way_home[current_node]
                        route_home.append(numpy.array(current_node))
                    q_list.insert(0 , q_start)
                dijkstras_results.append(g_values[1]) # shortest distance from start to goal to be plotted
                print "route to goal: ",route_home
            # - - - - - - - - - ^^ Dijkstras algorithm ^^ - - - - - - - - - - - 

            # - - - - - - - - - - Add new point to map - - - - - - - - - - - - -
            q_new=numpy.random.uniform(q_min, q_max) #random points
            while (not self.is_state_valid(q_new)):
                q_new=numpy.random.uniform(q_min, q_max)
            PRM_points.append(q_new)
            connections.append([])
            distances.append([])
            print "New q found.  Number of valid points on (PRM) map: ",len(PRM_points)
            # - - - - - - - - - ^^ Add new point to map ^^ - - - - - - - - - - - 

        #print "connections",connections
        #print "route home: ",route_home
        #print "q_goal =",q_goal
        #print "q_start =",q_start
        #print "all points: ",PRM_points
        #print "q_min = ",q_min
        #print "q_max = ",q_max
        #plt.plot(dijkstras_results)
        #plt.ylabel('Dijkstras shortest route (distance)')
        #plt.xlabel('Number of Dijkstras attempts')
        #plt.show()
        # - - use above lines of code to show how dijkstras result is improved over time - - 

        #print "q_list = ",q_list

        # - - - - - - - - - segment q_list - - - - - - - -
        if (1 in connected_nodes): 
            q_segmented = []
            q_segmented.append(q_list[0]) #Appending first point to sampled list
            q_seg=[]
            for i in range(0, len(q_list)-1):
                q_seg=q_list[i+1]-q_list[i] #Vector between k node and farthest nodes in path
                q_dir=q_seg/numpy.linalg.norm(q_seg) #unit vector from k node to farthest nodes       
                segments=int(math.floor(numpy.linalg.norm(q_seg)/0.5)) #Number of l=0.5 segments
                #print 'distance to sample ',numpy.linalg.norm(q_seg)
                #print 'number points in segment ', segments
                for k in range(0, segments):
                    q_new=q_list[i]+0.5*(k+1)*q_dir #k sampled point
                    if k==segments-1:
                        q_segmented.append(q_list[i+1]) #Adding in shortcut path instead of sampled point
                    else:
                        q_segmented.append(q_new) #Adding new sampled point to q_list
            q_list = q_segmented
            print 'Sampled path has ', len(q_list), ' nodes'
        else:
            print 'Path to goal not found.  Returning start and goal as only points'
            q_list = [q_start, q_goal]
        # - - - - - - ^^ segment q_list ^^ - - - - - - - - -
        

        return q_list    
        '--------------------------------------------------------------------------------------------------'



    def project_plan(self, q_start, q_goal, q_min, q_max):
        q_list, v_list, a_list, t = self.motion_plan(q_start, q_goal, q_min, q_max)
        joint_trajectory = self.create_trajectory(q_list, v_list, a_list, t)
        return joint_trajectory

    def moveit_plan(self, q_start, q_goal, q_min, q_max):
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(q_goal)
        plan=self.group.plan()
        joint_trajectory = plan.joint_trajectoryq_goal
        for i in range(0,len(joint_trajectory.points)):
            joint_trajectory.points[i].time_from_start = \
              rospy.Duration(joint_trajectory.points[i].time_from_start)
        return joint_trajectory        

    def create_trajectory(self, q_list, v_list, a_list, t):
        joint_trajectory = trajectory_msgs.msg.JointTrajectory()
        for i in range(0, len(q_list)):
            point = trajectory_msgs.msg.JointTrajectoryPoint()
            point.positions = list(q_list[i])
            point.velocities = list(v_list[i])
            point.accelerations = list(a_list[i])
            point.time_from_start = rospy.Duration(t[i])
            joint_trajectory.points.append(point)
        joint_trajectory.joint_names = self.joint_names
        return joint_trajectory

    def execute(self, joint_trajectory):
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory = joint_trajectory
        goal.goal_time_tolerance = rospy.Duration(0.0)
        self.trajectory_client.send_goal(goal)
        self.trajectory_client.wait_for_result()

    def sample_polynomial(self, coeffs, i, T):
        q = coeffs[4*i+0]*T*T*T + coeffs[4*i+1]*T*T + coeffs[4*i+2]*T + coeffs[4*i+3]
        v = coeffs[4*i+0]*3*T*T + coeffs[4*i+1]*2*T + coeffs[4*i+2]
        a = coeffs[4*i+0]*6*T   + coeffs[4*i+1]*2
        return (q,a,v)

    def plot_series(self, t_vec, y_vec, title):
        fig, ax = plt.subplots()
        line, = ax.plot(numpy.random.rand(10))
        ax.set_xlim(0, t_vec[-1])
        ax.set_ylim(min(y_vec),max(y_vec))
        line.set_xdata(deepcopy(t_vec))
        line.set_ydata(deepcopy(y_vec))
        fig.suptitle(title)

    def move_arm_cb(self, feedback):
        print 'Moving the arm'
        self.mutex.acquire()
        q_start = self.q_current
        T = convert_from_message(feedback.pose)
        print "Solving IK"
        q_goal = self.IK(T)
        if len(q_goal)==0:
            print "IK failed, aborting"
            self.mutex.release()
            return

        print "IK solved, planning"
        q_start = numpy.array(self.q_from_joint_state(self.joint_state))
        trajectory = self.project_plan(q_start, q_goal, self.q_min, self.q_max)
        if not trajectory.points:
            print "Motion plan failed, aborting"
        else:
            print "Trajectory received with " + str(len(trajectory.points)) + " points"
            self.execute(trajectory)
        self.mutex.release()

    def no_obs_cb(self, feedback):
        print 'Removing all obstacles'
        self.scene.remove_world_object("obs1")
        self.scene.remove_world_object("obs2")
        self.scene.remove_world_object("obs3")
        self.scene.remove_world_object("obs4")

    def simple_obs_cb(self, feedback):
        print 'Adding simple obstacle'
        self.no_obs_cb(feedback)
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.5, 0.5, 0)) )
        
        self.scene.add_box("obs1", pose_stamped,(0.1,0.1,1))

    def complex_obs_cb(self, feedback):
        print 'Adding hard obstacle'
        self.no_obs_cb(feedback)
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.5, 0.2)) )
        self.scene.add_box("obs1", pose_stamped,(0.1,0.1,0.8))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.25, 0.6)) )
        self.scene.add_box("obs2", pose_stamped,(0.1,0.5,0.1))

    def super_obs_cb(self, feedback):
        print 'Adding super hard obstacle'
        self.no_obs_cb(feedback)
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time(0)
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.5, 0.2)) )
        self.scene.add_box("obs1", pose_stamped,(0.1,0.1,0.8))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.25, 0.6)) )
        self.scene.add_box("obs2", pose_stamped,(0.1,0.5,0.1))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.0, 0.2)) )
        self.scene.add_box("obs3", pose_stamped,(0.1,0.1,0.8))
        pose_stamped.pose = convert_to_message( tf.transformations.translation_matrix((0.7, 0.25, 0.1)) )
        self.scene.add_box("obs4", pose_stamped,(0.1,0.5,0.1))


    def plot_cb(self,feedback):
        handle = feedback.menu_entry_id
        state = self.menu_handler.getCheckState( handle )
        if state == MenuHandler.CHECKED: 
            self.show_plots = False
            print "Not showing plots"
            self.menu_handler.setCheckState( handle, MenuHandler.UNCHECKED )
        else:
            self.show_plots = True
            print "Showing plots"
            self.menu_handler.setCheckState( handle, MenuHandler.CHECKED )
        self.menu_handler.reApply(self.server)
        self.server.applyChanges()
        
    def joint_states_callback(self, joint_state):
        self.mutex.acquire()
        self.q_current = joint_state.position
        self.joint_state = joint_state
        self.mutex.release()

    def init_marker(self):

        self.server = InteractiveMarkerServer("control_markers")

        control_marker = InteractiveMarker()
        control_marker.header.frame_id = "/base"
        control_marker.name = "move_arm_marker"

        move_control = InteractiveMarkerControl()
        move_control.name = "move_x"
        move_control.orientation.w = 1
        move_control.orientation.x = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "move_y"
        move_control.orientation.w = 1
        move_control.orientation.y = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "move_z"
        move_control.orientation.w = 1
        move_control.orientation.z = 1
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control_marker.controls.append(move_control)

        move_control = InteractiveMarkerControl()
        move_control.name = "rotate_x"
        move_control.orientation.w = 1
        move_control.orientation.x = 1
        move_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "rotate_y"
        move_control.orientation.w = 1
        move_control.orientation.z = 1
        move_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control_marker.controls.append(move_control)
        move_control = InteractiveMarkerControl()
        move_control.name = "rotate_z"
        move_control.orientation.w = 1
        move_control.orientation.y = 1
        move_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control_marker.controls.append(move_control)

        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.BUTTON
        menu_control.always_visible = True
        box = Marker()        
        box.type = Marker.CUBE
        box.scale.x = 0.15
        box.scale.y = 0.03
        box.scale.z = 0.03
        box.color.r = 0.5
        box.color.g = 0.5
        box.color.b = 0.5
        box.color.a = 1.0
        menu_control.markers.append(box)
        box2 = deepcopy(box)
        box2.scale.x = 0.03
        box2.scale.z = 0.1
        box2.pose.position.z=0.05
        menu_control.markers.append(box2)
        control_marker.controls.append(menu_control)

        control_marker.scale = 0.25        
        self.server.insert(control_marker, self.control_marker_feedback)

        self.menu_handler = MenuHandler()
        self.menu_handler.insert("Move Arm", callback=self.move_arm_cb)
        obs_entry = self.menu_handler.insert("Obstacles")
        self.menu_handler.insert("No Obstacle", callback=self.no_obs_cb, parent=obs_entry)
        self.menu_handler.insert("Simple Obstacle", callback=self.simple_obs_cb, parent=obs_entry)
        self.menu_handler.insert("Hard Obstacle", callback=self.complex_obs_cb, parent=obs_entry)
        self.menu_handler.insert("Super-hard Obstacle", callback=self.super_obs_cb, parent=obs_entry)
        options_entry = self.menu_handler.insert("Options")
        self.plot_entry = self.menu_handler.insert("Plot trajectory", parent=options_entry,
                                                     callback = self.plot_cb)
        self.menu_handler.setCheckState(self.plot_entry, MenuHandler.UNCHECKED)
        self.menu_handler.apply(self.server, "move_arm_marker",)

        self.server.applyChanges()

        Ttrans = tf.transformations.translation_matrix((0.6,0.2,0.2))
        Rtrans = tf.transformations.rotation_matrix(3.14159,(1,0,0))
        self.server.setPose("move_arm_marker", convert_to_message(numpy.dot(Ttrans,Rtrans)))
        self.server.applyChanges()


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_arm', anonymous=True)
    ma = MoveArm()
    rospy.spin()

