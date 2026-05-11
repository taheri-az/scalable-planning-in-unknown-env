#!/usr/bin/env python3
import math
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


class PID:
    """Standard PID with output saturation and integral anti-windup."""

    def __init__(self, kp, ki, kd, out_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self._prev_err = None
        self._prev_t = None

    def step(self, err, t):
        if self._prev_t is None:
            dt = 0.0
            d_err = 0.0
        else:
            dt = t - self._prev_t
            d_err = (err - self._prev_err) / dt if dt > 0 else 0.0

        self.integral += err * dt
        if self.ki > 0:
            i_max = self.out_limit / self.ki
            self.integral = max(-i_max, min(i_max, self.integral))

        out = (self.kp * err
               + self.ki * self.integral
               + self.kd * d_err)
        out = max(-self.out_limit, min(self.out_limit, out))

        self._prev_err = err
        self._prev_t = t
        return out


class TurtleBot:
    # Target world-frame yaw (radians) for each grid action.
    # Convention: +x = "right", +y = "up".
    HEADINGS = {
        'up':     math.pi / 2,
        'down':  -math.pi / 2,
        'left':   math.pi,
        'right':  0.0,
    }

    # Hard motion limits (TB3 Burger: 0.22 m/s, ~2.84 rad/s)
    LINEAR_SPEED    = 0.15
    ANGULAR_SPEED   = 1.5

    STEP_DISTANCE   = 0.20   # 20 cm per action

    # PID gains — tuned for TB3 Burger; tune for your bot if needed.
    KP_ANG, KI_ANG, KD_ANG = 3.0, 0.0, 0.3
    KP_LIN, KI_LIN, KD_LIN = 1.5, 0.0, 0.0

    ANGLE_TOLERANCE = 0.015   # ~0.86 deg
    DIST_TOLERANCE  = 0.005   # 5 mm
    MOTION_TIMEOUT  = 8.0     # seconds per move — safety cap

    def __init__(self, node_name='turtle_mover', start_facing='right'):
        rospy.init_node(node_name, anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._have_odom = False
        self.rate = rospy.Rate(20)

        time.sleep(1.0)
        while not self._have_odom and not rospy.is_shutdown():
            self.rate.sleep()

        if start_facing not in self.HEADINGS:
            raise ValueError(f"start_facing must be one of {list(self.HEADINGS)}")
        self.yaw_offset = self.yaw - self.HEADINGS[start_facing]

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.x, self.y = p.x, p.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._have_odom = True

    def _stop(self):
        self.pub.publish(Twist())

    def _rotate_to(self, target_yaw):
        pid = PID(self.KP_ANG, self.KI_ANG, self.KD_ANG, self.ANGULAR_SPEED)
        deadline = time.time() + self.MOTION_TIMEOUT
        while not rospy.is_shutdown():
            err = math.atan2(math.sin(target_yaw - self.yaw),
                             math.cos(target_yaw - self.yaw))
            if abs(err) < self.ANGLE_TOLERANCE or time.time() > deadline:
                break
            cmd = Twist()
            cmd.angular.z = pid.step(err, time.time())
            self.pub.publish(cmd)
            self.rate.sleep()
        self._stop()

    def _move_forward(self, distance):
        x0, y0 = self.x, self.y
        pid = PID(self.KP_LIN, self.KI_LIN, self.KD_LIN, self.LINEAR_SPEED)
        deadline = time.time() + self.MOTION_TIMEOUT
        while not rospy.is_shutdown():
            traveled = math.hypot(self.x - x0, self.y - y0)
            err = distance - traveled
            if err < self.DIST_TOLERANCE or time.time() > deadline:
                break
            cmd = Twist()
            v = pid.step(err, time.time())
            cmd.linear.x = max(0.0, v)  # never reverse on overshoot
            self.pub.publish(cmd)
            self.rate.sleep()
        self._stop()

    def move(self, action):
        if action == 'stay':
            return
        if action not in self.HEADINGS:
            rospy.logwarn("Unknown action: %s", action)
            return

        target = math.atan2(
            math.sin(self.HEADINGS[action] + self.yaw_offset),
            math.cos(self.HEADINGS[action] + self.yaw_offset),
        )
        self._rotate_to(target)
        time.sleep(0.2)
        self._move_forward(self.STEP_DISTANCE)
        time.sleep(0.2)


if __name__ == '__main__':
    bot = TurtleBot()
    for a in ['up', 'right', 'down', 'left']:
        print(f"Action: {a}")
        bot.move(a)
    print('Done.')
