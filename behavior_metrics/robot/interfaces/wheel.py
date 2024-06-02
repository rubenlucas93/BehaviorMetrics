import rospy
from std_msgs.msg import Float32
import threading


class ListenerWheelAngle:
    '''
    ROS Wheel Angle Subscriber. Wheel Angle Client to receive wheel angle from ROS nodes.
    '''
    def __init__(self, topic):
        '''
        ListenerWheelAngle Constructor.

        @param topic: ROS topic to subscribe
        @type topic: String
        '''
        self.topic = topic
        self.data = 0.0
        self.sub = None
        self.lock = threading.Lock()
        self.start()

    def __callback(self, wheel_angle):
        '''
        Callback function to receive and save wheel angle.

        @param wheel_angle: ROS message with wheel angle

        @type wheel_angle: Float32
        '''
        self.lock.acquire()
        self.data = wheel_angle.data
        self.lock.release()

    def stop(self):
        '''
        Stops (Unregisters) the client.
        '''
        self.sub.unregister()

    def start(self):
        '''
        Starts (Subscribes) the client.
        '''
        self.sub = rospy.Subscriber(self.topic, Float32, self.__callback)

    def getWheelAngle(self):
        '''
        Returns last wheel angle received.
        @return last wheel angle saved
        '''
        self.lock.acquire()
        wheel_angle = self.data
        self.lock.release()
        return wheel_angle
