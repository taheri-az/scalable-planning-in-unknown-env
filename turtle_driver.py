#!/usr/bin/env python3
import math
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


class TurtleBot:
    # Target world-frame yaw (radians) for each grid action.
    # Convention: +x = "right" (east), +y = "up" (north / top).
    HEADINGS = {
        'up':     math.pi / 2,
        'down':  -math.pi / 2,
        'left':   math.pi,
        'right':  0.0,
    }

    LINEAR_SPEED    = 0.1   # m/s
    ANGULAR_SPEED   = 0.5   # rad/s
    STEP_DISTANCE   = 0.20  # 20 cm per action
    ANGLE_TOLERANCE = 0.02  # ~1.1 deg

    def __init__(self, node_name='turtle_mover'):
        rospy.init_node(node_name, anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        self.yaw = 0.0
        self._have_odom = False
        self.rate = rospy.Rate(20)

        time.sleep(1.0)  # let the publisher register with /cmd_vel
        while not self._have_odom and not rospy.is_shutdown():
            self.rate.sleep()

    def _odom_cb(self, msg):
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._have_odom = True

    def _stop(self):
        self.pub.publish(Twist())

    def _rotate_to(self, target_yaw):
        while not rospy.is_shutdown():
            err = math.atan2(math.sin(target_yaw - self.yaw),
                             math.cos(target_yaw - self.yaw))
            if abs(err) < self.ANGLE_TOLERANCE:
                break
            cmd = Twist()
            speed = self.ANGULAR_SPEED
            if abs(err) < 0.3:
                speed *= max(abs(err) / 0.3, 0.2)  # ease in near target
            cmd.angular.z = speed if err > 0 else -speed
            self.pub.publish(cmd)
            self.rate.sleep()
        self._stop()

    def _move_forward(self, distance):
        duration = distance / self.LINEAR_SPEED
        cmd = Twist()
        cmd.linear.x = self.LINEAR_SPEED
        start = time.time()
        while time.time() - start < duration and not rospy.is_shutdown():
            self.pub.publish(cmd)
            self.rate.sleep()
        self._stop()

    def move(self, action):
        """Adjust heading for `action`, then drive 20 cm forward."""
        if action == 'stay':
            return
        if action not in self.HEADINGS:
            rospy.logwarn("Unknown action: %s", action)
            return

        self._rotate_to(self.HEADINGS[action])
        time.sleep(0.2)
        self._move_forward(self.STEP_DISTANCE)
        time.sleep(0.2)


if __name__ == '__main__':
    bot = TurtleBot()
    for a in ['up', 'right', 'down', 'left']:
        print(f"Action: {a}")
        bot.move(a)
    print('Done.')
