import numpy as np

class PID:
    def __init__(self, P_coeff, I_coeff, D_coeff, goal):
        self.P_coeff = P_coeff
        self.I_coeff = I_coeff
        self.D_coeff = D_coeff
        self.integral = 0
        self.last_error = 0
        self.goal = goal

    def observe(self, x):
        error = self.goal - x
        d_error = error - self.last_error
        self.last_error = error
        self.integral += error
        u_ = self.P_coeff * error + self.I_coeff * self.integral + self.D_coeff * d_error
        return u_

class Controller:
    def __init__(self):
        self.pole_controller = PID(5,0,100, 0)
        self.cart_controller = PID(1,0,100, 0)

    def observe(self, observation):
        pole_angle = observation[2]
        u_pole = self.pole_controller.observe(pole_angle)
        cart_position = observation[0]
        u_cart = self.cart_controller.observe(cart_position)
        action = 1 if u_pole + u_cart < 0 else 0
        return action

class MountainCarController:
    def __init__(self):
        self.state_dim = 2
        self.action_dim = 1

    def observe(self, observation) -> np.ndarray:
        position = observation[0]
        velocity = observation[1]
        
        # More aggressive strategy for continuous mountain car
        # Focus on building maximum momentum and timing the final push
        
        if position < -0.5:
            # When far left, push right with maximum force
            if velocity < 0:
                # If moving left, push right hard to build momentum
                action = -1.0
            else:
                # If moving right, keep pushing hard
                action = 1.0
        elif position > 0.4:
            # When near the goal, use momentum and gravity
            if velocity > 0:
                # If moving right with good momentum, give a final push
                action = 0.8
            else:
                # If moving left, let gravity help build momentum
                action = -0.5
        else:
            # In the middle region
            if velocity > 0:
                # If moving right, maintain momentum
                action = 1.0
            else:
                # If moving left, push hard to reverse direction
                action = -1.0
        return np.array([action])
