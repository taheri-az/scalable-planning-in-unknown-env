#!/usr/bin/env python3
import math
import threading
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
    LINEAR_SPEED    = 0.05
    ANGULAR_SPEED   = 1.5

    # Linear velocity held during a heading change. Nonzero so the robot
    # arcs through corners instead of pivoting in place.
    TURN_LINEAR_SPEED = 0.04

    # Physical cell size — robot traverses CELL_SIZE meters between cell centers.
    CELL_SIZE       = 0.3

    # PID gains — tuned for TB3 Burger; tune for your bot if needed.
    KP_ANG, KI_ANG, KD_ANG = 3.0, 0.0, 0.3
    KP_LIN, KI_LIN, KD_LIN = 1.5, 0.0, 0.0

    ANGLE_TOLERANCE = 0.015   # ~0.86 deg
    DIST_TOLERANCE  = 0.005   # 5 mm
    MOTION_TIMEOUT  = 12.0    # seconds per cell — safety cap (slower speed needs more headroom)

    def __init__(self, node_name='turtle_mover', start_facing='right'):
        rospy.init_node(node_name, anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('/odom', Odometry, self._odom_cb)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._have_odom = False
        self.rate = rospy.Rate(20)

        while not self._have_odom and not rospy.is_shutdown():
            self.rate.sleep()

        if start_facing not in self.HEADINGS:
            raise ValueError(f"start_facing must be one of {list(self.HEADINGS)}")
        self.yaw_offset = self.yaw - self.HEADINGS[start_facing]

        # Threaded-motion state.
        self._lock = threading.Lock()
        self._next_action = None              # single-slot queue (one action lookahead)
        self._next_action_event = threading.Event()
        self._cell_entered = threading.Event()  # set when robot crosses next-cell boundary
        self._idle = threading.Event()
        self._idle.set()
        self._current_yaw_target = None       # heading the robot last drove toward
        self._cell_start_xy = None            # (x, y) at the start of the current cell traversal
        self._shutdown = False

        self._motion_thread = threading.Thread(target=self._motion_loop, daemon=True)
        self._motion_thread.start()

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.x, self.y = p.x, p.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._have_odom = True

    def _stop(self):
        self.pub.publish(Twist())

    def _action_target_yaw(self, action):
        return math.atan2(
            math.sin(self.HEADINGS[action] + self.yaw_offset),
            math.cos(self.HEADINGS[action] + self.yaw_offset),
        )

    @staticmethod
    def _yaw_diff(a, b):
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

    def _blend_rotate(self, target_yaw):
        """Rotate to `target_yaw` while keeping a slow forward velocity, so
        the robot arcs through the corner rather than stopping to pivot."""
        pid = PID(self.KP_ANG, self.KI_ANG, self.KD_ANG, self.ANGULAR_SPEED)
        deadline = time.time() + self.MOTION_TIMEOUT
        while not rospy.is_shutdown() and not self._shutdown:
            err = math.atan2(math.sin(target_yaw - self.yaw),
                             math.cos(target_yaw - self.yaw))
            if abs(err) < self.ANGLE_TOLERANCE or time.time() > deadline:
                break
            cmd = Twist()
            cmd.angular.z = pid.step(err, time.time())
            cmd.linear.x = self.TURN_LINEAR_SPEED
            self.pub.publish(cmd)
            self.rate.sleep()
        # No _stop() here — let _drive_continuous take over the cmd_vel stream
        # so there's no zero-velocity gap between rotation and the next cell.

    def _drive_continuous(self, x0, y0, target_yaw):
        """
        Drive forward along `target_yaw` starting from (x0, y0). Fire
        `_cell_entered` at the half-cell mark. After half-cell, if a queued
        next action exists in the same direction, consume it and extend the
        target by one more cell — no deceleration at the boundary. Exit when
        the cell is complete and no same-direction follow-up is available.
        """
        cos_h = math.cos(target_yaw)
        sin_h = math.sin(target_yaw)
        self._cell_start_xy = (x0, y0)
        self._cell_entered.clear()
        signaled = False
        pid = PID(self.KP_LIN, self.KI_LIN, self.KD_LIN, self.LINEAR_SPEED)
        deadline = time.time() + self.MOTION_TIMEOUT

        while not rospy.is_shutdown() and not self._shutdown:
            # Signed distance projected onto the heading direction — supports
            # extending the target while the robot is still mid-cell.
            traveled = (self.x - x0) * cos_h + (self.y - y0) * sin_h
            err = self.CELL_SIZE - traveled

            if not signaled and traveled >= self.CELL_SIZE / 2:
                self._cell_entered.set()
                signaled = True

            # Only chain after we've delivered the halfway signal for this
            # cell, otherwise we'd swallow it.
            if signaled:
                chained = False
                with self._lock:
                    na = self._next_action
                    if (na is not None
                        and na in self.HEADINGS
                        and self._yaw_diff(self._action_target_yaw(na), target_yaw)
                            < self.ANGLE_TOLERANCE):
                        self._next_action = None
                        self._next_action_event.clear()
                        chained = True
                if chained:
                    x0 += cos_h * self.CELL_SIZE
                    y0 += sin_h * self.CELL_SIZE
                    self._cell_start_xy = (x0, y0)
                    self._cell_entered.clear()
                    signaled = False
                    deadline = time.time() + self.MOTION_TIMEOUT
                    continue

            if err < self.DIST_TOLERANCE or time.time() > deadline:
                break

            cmd = Twist()
            cmd.linear.x = max(0.0, pid.step(err, time.time()))
            self.pub.publish(cmd)
            self.rate.sleep()

        if not signaled:
            # Guarantee progress in pathological cases (timeout, shutdown).
            self._cell_entered.set()

    def _execute_action(self, action):
        if action == 'stay':
            self._cell_entered.set()
            return
        if action not in self.HEADINGS:
            rospy.logwarn("Unknown action: %s", action)
            self._cell_entered.set()
            return

        target_yaw = self._action_target_yaw(action)
        same_direction = (
            self._current_yaw_target is not None
            and self._yaw_diff(target_yaw, self._current_yaw_target) < self.ANGLE_TOLERANCE
        )

        if not same_direction:
            self._blend_rotate(target_yaw)
            self._current_yaw_target = target_yaw
            x0, y0 = self.x, self.y
        else:
            # Same direction — start the new cell from where the previous cell
            # ended, even if the robot is still in motion. This keeps absolute
            # cell positions exact and avoids drift accumulation.
            if self._cell_start_xy is not None:
                cos_h = math.cos(target_yaw)
                sin_h = math.sin(target_yaw)
                x0 = self._cell_start_xy[0] + cos_h * self.CELL_SIZE
                y0 = self._cell_start_xy[1] + sin_h * self.CELL_SIZE
            else:
                x0, y0 = self.x, self.y

        self._drive_continuous(x0, y0, target_yaw)

    def _motion_loop(self):
        while not rospy.is_shutdown() and not self._shutdown:
            self._next_action_event.wait()
            if self._shutdown:
                break
            with self._lock:
                action = self._next_action
                self._next_action = None
                self._next_action_event.clear()
            if action is None:
                continue

            self._execute_action(action)

            # If nothing else queued, fully stop and mark idle.
            with self._lock:
                follow_up = self._next_action
            if follow_up is None:
                self._stop()
                self._idle.set()
        self._stop()

    # ---------- public API ----------

    def move(self, action):
        """Queue the next action. Returns immediately; motion runs in the background."""
        with self._lock:
            if self._next_action is not None:
                rospy.logwarn(
                    "Overwriting queued action %s with %s "
                    "(main is queueing faster than motion can chain)",
                    self._next_action, action,
                )
            self._next_action = action
        self._cell_entered.clear()
        self._idle.clear()
        self._next_action_event.set()

    def wait_for_cell_entry(self):
        """Block until the robot has crossed into the next cell (halfway through CELL_SIZE)."""
        self._cell_entered.wait()

    def wait_for_heading_settled(self, tolerance=0.08, timeout=1.5):
        """Block until the robot's actual yaw is within `tolerance` of the
        last commanded heading, or `timeout` seconds elapses. Useful right
        before reading the camera so the FOV is aligned with the action
        direction (and not mid-turn)."""
        if self._current_yaw_target is None:
            return
        deadline = time.time() + timeout
        while not rospy.is_shutdown() and not self._shutdown and time.time() < deadline:
            err = math.atan2(
                math.sin(self._current_yaw_target - self.yaw),
                math.cos(self._current_yaw_target - self.yaw),
            )
            if abs(err) < tolerance:
                return
            time.sleep(0.02)

    def wait(self):
        """Block until all queued motion has finished and the robot is stopped."""
        self._idle.wait()

    def shutdown(self):
        self._shutdown = True
        # Wake every blocked waiter so threads exit promptly.
        self._next_action_event.set()
        self._cell_entered.set()
        self._idle.set()
        self._stop()


if __name__ == '__main__':
    bot = TurtleBot()
    for a in ['up', 'right', 'down', 'left']:
        print(f"Action: {a}")
        bot.move(a)
        bot.wait_for_cell_entry()
    bot.wait()
    print('Done.')
