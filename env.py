import cv2
import numpy as np

ENV_SIZE = 72


class ArmEnvironment(object):
	def __init__(self, n_joints=2, max_vel=np.pi / 4, time_lim=16):
		super(ArmEnvironment, self).__init__()
		self.w = ENV_SIZE
		self.h = ENV_SIZE
		self.center = int(self.w / 2), int(self.h / 2)
		self.img = np.zeros((self.h, self.w, 3), np.uint8)
		self.overlay = np.zeros((self.h, self.w, 3), np.uint8)

		self.time_lim = time_lim

		self.n_joints = n_joints
		self.colors = [255 * (i / (self.n_joints - 1)) for i in range(self.n_joints)]
		self.action_bounds = [-1] * self.n_joints, [1] * self.n_joints
		self.max_vel = max_vel
		self.vel_bounds = [-self.max_vel] * self.n_joints, [self.max_vel] * self.n_joints
		self.segment_l = int(min(self.w, self.h) / 2 / self.n_joints)

		self.joint_states = [0] * self.n_joints

		self.is_terminated = self.action = self.time = self.val = self.err_sum = None
		self.reset()

	def close(self):
		pass

	def render(self, imshow=True):
		# Clear
		# cv2.rectangle(self.overlay, (0, 0), (self.w, self.h), (0, 0, 0, 255), cv2.FILLED)

		# Segments
		start = self.center
		abs_states = np.cumsum(self.joint_states)
		col = self.time / self.time_lim
		for sin, cos, color in zip(np.sin(abs_states), np.cos(abs_states), self.colors):
			end = (start[0] + int(self.segment_l * cos), start[1] + int(self.segment_l * sin))
			cv2.line(self.img, start, end, (255 * (1 - col), color, 255 * col), 1)
			start = end

		# alpha = 1.0 / self.time_lim
		# cv2.addWeighted(self.overlay, alpha, self.img, 1 - alpha, .1, dst=self.img)

		if imshow:
			cv2.imshow('Arm Environment', self.img)
			cv2.waitKey(1)

	def reset(self):
		self.is_terminated = False
		self.action = [0, 0]
		self.joint_states = np.random.uniform(-np.pi / 2, np.pi / 2, self.n_joints)
		self.time = 0
		self.val = 0
		self.err_sum = 0
		cv2.rectangle(self.img, (0, 0), (self.w, self.h), (0, 0, 0, 255), cv2.FILLED)
		return np.array(self.joint_states)  # , False, None, None

	def goal_range(self):
		return

	def step(self, action):
		self.time += 1
		if self.time > 80:
			# print("Value:", self.val)
			if self.is_terminated:
				raise RuntimeError("The episode has already terminated, please reset the environment")
			self.is_terminated = True

		self.joint_states = action
		return np.array(self.joint_states), 0, self.is_terminated, None


class ArmTorqueEnvironment(ArmEnvironment):
	def step(self, action):
		action = np.clip(action, *self.action_bounds)
		self.action = action
		joint_vels = np.clip(self.max_vel * action, *self.vel_bounds)
		self.joint_states += joint_vels
		return super(ArmTorqueEnvironment, self).step(self.joint_states)


class MultiArmEnvironmentBase(object):
	def __init__(self, n_arms=4, n_joints=2, max_vel=np.pi / 4, time_lim=16, sub_env_type=ArmEnvironment):
		super(MultiArmEnvironmentBase, self).__init__()

		self.n_arms = n_arms
		self.n_w = int(np.sqrt(self.n_arms))
		self.n_h = int(np.ceil(self.n_arms / self.n_w))
		self.env_size = ENV_SIZE
		self.w = self.env_size * self.n_w
		self.h = self.env_size * self.n_h
		self.img = np.zeros((self.h, self.w, 3), np.uint8)
		self.envs = []
		for i in range(self.n_arms):
			x_i = i % self.n_w
			y_i = int(i / self.n_w)
			env = sub_env_type(n_joints, max_vel, time_lim)
			env.img = self.img[
					  x_i * self.env_size:(x_i + 1) * self.env_size,
					  y_i * self.env_size:(y_i + 1) * self.env_size]
			self.envs.append(env)
		self.is_terminated = self.time = None
		self.reset()

	def close(self):
		pass

	def render(self):
		for env in self.envs:
			env.render(False)
		cv2.imshow('Arm Environment', self.img)
		cv2.waitKey(1)

	def reset(self):
		self.is_terminated = False
		self.time = 0
		return [env.reset() for env in self.envs]

	def step(self, actions):
		self.time += 1
		if self.time > 80:
			# print("Value:", self.val)
			if self.is_terminated:
				raise RuntimeError("The episode has already terminated, please reset the environment")
			self.is_terminated = True
		results = [env.step(action) for env, action in zip(self.envs, actions)]
		results = [np.array(result[i] for result in results) for i in range(4)]

		return results


class MultiArmEnvironment(MultiArmEnvironmentBase):
	def __init__(self, n_arms=4, n_joints=2, max_vel=np.pi / 4, time_lim=16):
		super(MultiArmEnvironment, self).__init__(n_arms, n_joints, max_vel, time_lim, ArmEnvironment)


class MultiArmTorqueEnvironment(MultiArmEnvironmentBase):
	def __init__(self, n_arms=4, n_joints=2, max_vel=np.pi / 4, time_lim=16):
		super(MultiArmTorqueEnvironment, self).__init__(n_arms, n_joints, max_vel, time_lim, ArmEnvironment)
