import pyray
import numpy as np
from numpy.random import RandomState
import os
import threading
from gbp.gbp import Factor
from gbp.gbp import VariableNode
from gbp.factors import linear_displacement


class game:
    def __init__(self, graph=None, n_rand=500):
        self.prng = RandomState(69)
        # OPTIONS --------------------
        self.click_to_add = True
        self.b_run = False
        self.b_wild = False
        self.b_GBP_scenario = False
        self.b_pyamg = False
        self.IPU_process_running = False
        self.read_event = threading.Event()
        self.write_event = threading.Event()
        self.reset_event = threading.Event()
        self.pause_event = threading.Event()
        self.skip_event = threading.Event()
        if graph is not None:
            self.graph = graph
            self.b_multi = graph.multigrid
        self.b_show_active = False
        
        self.C = []
        self.C_var_ids = []
        self.C_base_ids = []
        self.layer_factor_ids = []
        self.show_layer = 0
        
        self.node_size = 5
        self.txt_off = int(self.node_size/2)
        self.scale = 1
        self.SCREEN_WIDTH = 1800
        self.SCREEN_HEIGHT = 900

        pyray.set_target_fps(30)

        self.camera = pyray.Camera2D()
        self.camera.zoom = 1.0
        self.ZOOM_INCREMENT = 0.125

        self.means = []
        self.connections = []
        self.n_vars = 0
        self.n_factors = 0
        self.n_rand = n_rand
        self.factors = []
        self.var_nodes = []

        self.meas_noise = 5.
        self.meas_noise_trans = 2.
        self.meas_noise_rot = 1.
        self.lambda_prior = 1e-8
        self.lambda_anchor = 1e8

        self.agent_pose = np.array([200., 200., 0.])
        self.agent_pose_history = [self.agent_pose.copy()]
        new_var_node = VariableNode(self.n_vars, 2)
        new_var_node.prior.eta = (self.agent_pose[0:2].copy()) * self.lambda_anchor
        new_var_node.prior.lam = np.diag([self.lambda_anchor, self.lambda_anchor])
        new_var_node.belief.eta = new_var_node.prior.eta.copy()
        new_var_node.belief.lam = new_var_node.prior.lam.copy()
        new_var_node.Sigma = 1/np.diagonal(new_var_node.belief.lam)
        new_var_node.mu = new_var_node.Sigma * new_var_node.belief.eta
        new_var_node.type = "pose"
        new_var_node.GT = self.agent_pose[0:2].copy()

        self.var_nodes.append(new_var_node)
        self.n_vars += 1
    
        self.var_theta_noisy = [self.agent_pose[2]]

        self.last_pose_var = self.var_nodes[-1]

        self.agent_radius = 150
        self.agent_angle = 360
        self.agent_vel = 5
        self.agent_ang_vel = 5
        self.pose_delta = np.array([0., 0., 0.])

        self.landmark_pos = []
        self.landmark_vars = []
        self.n_lvars = 0

        self.r_nvars = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, 30*2, 100*2, 30*2)
        self.r_random = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, 70*2, 100*2, 30*2)
        self.r_GBP = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, 150*2, 100*2, 30*2)
        self.r_clear = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, 110*2, 100*2, 30*2)
        self.r_save = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, self.SCREEN_HEIGHT - 40*2, 100*2, 30*2)
        self.r_graph = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, self.SCREEN_HEIGHT - 80*2, 100*2, 30*2)
        self.r_reset = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, 10, 100*2, 30*2)
        self.r_wildtoggle = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, self.SCREEN_HEIGHT - 120*2, 100*2, 30*2)
        self.r_pyamgtoggle = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, self.SCREEN_HEIGHT - 160*2, 100*2, 30*2)
        self.r_multitoggle = pyray.Rectangle(self.SCREEN_WIDTH - 110*2, self.SCREEN_HEIGHT - 200*2, 100*2, 30*2)

        self.r_play = pyray.Rectangle(10, self.SCREEN_HEIGHT- 60, 50, 50)
        self.r_pause = pyray.Rectangle(60, self.SCREEN_HEIGHT- 60, 50, 50)
        self.r_skip = pyray.Rectangle(120, self.SCREEN_HEIGHT- 60, 50, 50)

        self.r_layer0 = pyray.Rectangle(420, self.SCREEN_HEIGHT- 60, 90, 50)
        self.r_layer1 = pyray.Rectangle(520, self.SCREEN_HEIGHT- 60, 90, 50)
        self.r_layer2 = pyray.Rectangle(620, self.SCREEN_HEIGHT- 60, 90, 50)
        self.r_layer3 = pyray.Rectangle(720, self.SCREEN_HEIGHT- 60, 90, 50)
        self.r_layer4 = pyray.Rectangle(820, self.SCREEN_HEIGHT- 60, 90, 50)
        self.r_layer_active = pyray.Rectangle(1020, self.SCREEN_HEIGHT- 60, 180, 50)

        self.c_true = pyray.Color(0,255,0,50)
        self.c_false = pyray.Color(255,0,0,50)
        self.c_orange = pyray.Color(255,165,0,50)


    def run(self):
        pyray.init_window(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, "2D Posegraph parallel GBP")
        pyray.gui_set_style(0,16,18)

        self.t_play = pyray.load_texture(os.getcwd() + "/resources/play_5050.png")
        self.t_skip = pyray.load_texture(os.getcwd() + "/resources/skip_5050.png")
        self.t_pause = pyray.load_texture(os.getcwd() + "/resources/pause_5050.png")
        self.t_agent = pyray.load_texture(os.getcwd() + "/resources/robot.png")
        self.t_landmark = pyray.load_texture(os.getcwd() + "/resources/star.png")
        # self.t_background = pyray.load_texture(os.getcwd() + "/resources/space.png")

        while not pyray.window_should_close():   
            # update
            self.user_input()
            # draw
            pyray.begin_drawing()
            pyray.clear_background(pyray.BLACK)
            pyray.begin_mode_2d(self.camera)

            # self.draw_background()
            self.draw_landmarks()
            self.draw_nodes()
            self.draw_agent()

            pyray.end_mode_2d()

            self.draw_interfaces()
            self.draw_textures()

            pyray.end_drawing()

    # def draw_background(self):

    #     cam_pos = self.camera.target
    #     n_height_tiles = np.ceil(self.SCREEN_HEIGHT/self.t_background.height).astype(int)
    #     n_width_tiles = np.ceil(self.SCREEN_WIDTH/self.t_background.width).astype(int)

    #     tile_x = np.array([*range(0, (n_width_tiles+2) * self.t_background.width, self.t_background.width)]) + np.floor(cam_pos.x / self.t_background.width) * self.t_background.width
    #     tile_y = np.array([*range(0, (n_height_tiles+2) * self.t_background.height, self.t_background.height)]) + np.floor(cam_pos.y / self.t_background.height) * self.t_background.height

    #     X, Y = np.meshgrid(tile_x, tile_y)
    #     tile_pos = np.vstack([X.ravel(), Y.ravel()])
    #     for pos in tile_pos.T:
    #         pyray.draw_texture_ex(self.t_background, pyray.Vector2(pos[0] - self.t_background.width/2, pos[1] - self.t_background.height/2),
    #                 0.0, 1, pyray.RAYWHITE)
        


    def draw_landmarks(self):
        for id in range(len(self.landmark_pos)):
            pyray.draw_texture_ex(self.t_landmark, pyray.Vector2(self.landmark_pos[id][0] - self.t_landmark.width/40, self.SCREEN_HEIGHT- self.landmark_pos[id][1] + - self.t_landmark.height/40),
                                  0.0, 0.05, pyray.RAYWHITE)
                

    def draw_agent(self):
        pyray.draw_circle_sector(pyray.Vector2(self.agent_pose[0], self.SCREEN_HEIGHT- self.agent_pose[1]), self.agent_radius, 
                                 self.agent_pose[2]-self.agent_angle/2+90, self.agent_pose[2]+self.agent_angle/2+90, 0, pyray.Color(0,255,0,50))
        pyray.draw_circle_sector_lines(pyray.Vector2(self.agent_pose[0], self.SCREEN_HEIGHT- self.agent_pose[1]), self.agent_radius, 
                                 self.agent_pose[2]-self.agent_angle/2+90, self.agent_pose[2]+self.agent_angle/2+90, 0, pyray.Color(0,255,0,255))
        pyray.draw_texture_pro(self.t_agent, pyray.Rectangle(0, 0, self.t_agent.width, self.t_agent.height), 
                               pyray.Rectangle(self.agent_pose[0], self.SCREEN_HEIGHT- self.agent_pose[1], self.t_agent.width/20, self.t_agent.height/20),
                               pyray.Vector2(self.t_agent.width/40, self.t_agent.height/40), -self.agent_pose[2]-90, pyray.RAYWHITE)
        
    
    def draw_interfaces(self):
            #pyray.gui_text_input_box(self.r_nvars,"","","","",0,0)
            pyray.gui_button(self.r_random, "GENERATE RANDOM")
            pyray.gui_button(self.r_GBP, "MEANING OF LIFE")
            pyray.gui_button(self.r_clear, "CLEAR NODES")
            pyray.gui_button(self.r_save, "SAVE TO FILE")
            pyray.gui_button(self.r_graph, "RUN GRAPH")
            pyray.gui_button(self.r_reset, "RESET")
            pyray.gui_button(self.r_wildtoggle, "TOGGLE WILDFIRE")
            pyray.gui_button(self.r_pyamgtoggle, "USE PyAMG")
            pyray.gui_button(self.r_multitoggle, "TOGGLE MULTIGRID")
            pyray.draw_text("LAYERS:", 250, self.SCREEN_HEIGHT - 50, 32, pyray.WHITE)
            pyray.gui_button(self.r_layer0, " ALL ")
            pyray.gui_button(self.r_layer1, " 1 ")
            pyray.gui_button(self.r_layer2, " 2 ")
            pyray.gui_button(self.r_layer3, " 3 ")
            pyray.gui_button(self.r_layer4, " 4 ")
            pyray.gui_button(self.r_layer_active, " ACTIVE ")

            pyray.draw_fps(10,10)

    def draw_textures(self):
        pyray.draw_texture(self.t_play, 10, self.SCREEN_HEIGHT- 60, pyray.WHITE)
        pyray.draw_texture(self.t_pause, 60, self.SCREEN_HEIGHT- 60, pyray.WHITE)
        pyray.draw_texture(self.t_skip, 120, self.SCREEN_HEIGHT- 60, pyray.WHITE)

        if self.b_run:
            if self.pause_event.is_set():
                pyray.draw_rectangle_rec(self.r_graph, self.c_orange)
            else:
                pyray.draw_rectangle_rec(self.r_graph, self.c_true)

        if self.b_multi:
            pyray.draw_rectangle_rec(self.r_multitoggle,self.c_true)
        else:
            pyray.draw_rectangle_rec(self.r_multitoggle,self.c_false)

        if self.b_pyamg:
            pyray.draw_rectangle_rec(self.r_pyamgtoggle,self.c_true)
        else:
            pyray.draw_rectangle_rec(self.r_pyamgtoggle,self.c_false)


        if self.b_wild:
            pyray.draw_rectangle_rec(self.r_wildtoggle,self.c_true)
        else:
            pyray.draw_rectangle_rec(self.r_wildtoggle,self.c_false)

        if self.b_show_active:
            pyray.draw_rectangle_rec(self.r_layer_active, self.c_true)
        else:
            match self.show_layer:
                case 0:
                    pyray.draw_rectangle_rec(self.r_layer0, self.c_true)
                case 1:
                    pyray.draw_rectangle_rec(self.r_layer1, self.c_true)
                case 2:
                    pyray.draw_rectangle_rec(self.r_layer2, self.c_true)
                case 3:
                    pyray.draw_rectangle_rec(self.r_layer3, self.c_true)
                case 4:
                    pyray.draw_rectangle_rec(self.r_layer4, self.c_true)
            

    def user_input(self):
        if pyray.is_mouse_button_pressed(pyray.MOUSE_BUTTON_LEFT):
            mouseXY = pyray.get_mouse_position()

            if mouseXY.x < self.SCREEN_WIDTH and mouseXY.y < self.SCREEN_HEIGHT:
                if pyray.check_collision_point_rec(mouseXY, self.r_save):
                    self.save_data()
                elif pyray.check_collision_point_rec(mouseXY, self.r_graph) or pyray.check_collision_point_rec(mouseXY, self.r_play):
                    if self.n_vars != 0:
                        self.b_run = True
                        self.pause_event.clear()
                elif pyray.check_collision_point_rec(mouseXY, self.r_pause):
                    self.pause_event.set()
                elif pyray.check_collision_point_rec(mouseXY, self.r_skip):
                    if self.n_vars != 0:
                        self.skip_event.set()
                        self.pause_event.clear()
                        self.b_run = True
                elif pyray.check_collision_point_rec(mouseXY, self.r_random):
                    self.generate_random_landmarks()
                elif pyray.check_collision_point_rec(mouseXY, self.r_clear):
                    self.clear_nodes()
                elif pyray.check_collision_point_rec(mouseXY, self.r_GBP):
                    self.generate_meaning_of_life()
                elif pyray.check_collision_point_rec(mouseXY, self.r_reset):
                    self.reset()
                elif pyray.check_collision_point_rec(mouseXY, self.r_wildtoggle):
                    self.b_wild = bool(1 - int(self.b_wild))
                    print("WILDFIRE TOGGLED: {}".format(self.b_wild))
                elif pyray.check_collision_point_rec(mouseXY, self.r_multitoggle):
                    self.b_multi = bool(1 - int(self.b_multi))
                    print("MULTIGRID TOGGLED: {}".format(self.b_multi))
                elif pyray.check_collision_point_rec(mouseXY, self.r_pyamgtoggle):
                    self.b_pyamg = bool(1 - int(self.b_pyamg))
                    print("PYAMG TOGGLED: {}".format(self.b_pyamg))
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer0):
                    self.show_layer = 0
                    self.b_show_active = False
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer1):
                    self.show_layer = 1
                    self.b_show_active = False
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer2):
                    self.show_layer = 2
                    self.b_show_active = False
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer3):
                    self.show_layer = 3
                    self.b_show_active = False
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer4):
                    self.show_layer = 4
                    self.b_show_active = False
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer_active):
                    self.b_show_active = bool(1- int(self.b_show_active))
                elif self.click_to_add and not self.b_run:
                    mouseWorldPos = pyray.get_screen_to_world_2d(pyray.get_mouse_position(), self.camera)
                    self.add_landmark(mouseWorldPos.x, mouseWorldPos.y)
                # self.node_size = 15 - min(5 * (self.n_vars/100),10)

        # key_pressed = pyray.get_key_pressed()
        # if key_pressed != 0:
        #     print(key_pressed)
        
        if pyray.is_mouse_button_down(pyray.MOUSE_BUTTON_RIGHT):
            delta = pyray.get_mouse_delta()
            delta = pyray.vector2_scale(delta, -1.0 / self.camera.zoom)
            self.camera.target = pyray.vector2_add(self.camera.target, delta)

        # zoom based on mouse wheel
        wheel = pyray.get_mouse_wheel_move()
        if wheel != 0:
            mouseWorldPos = pyray.get_screen_to_world_2d(pyray.get_mouse_position(), self.camera)
            self.camera.offset = pyray.get_mouse_position()
            self.camera.target = mouseWorldPos
            
            self.camera.zoom += (wheel*self.ZOOM_INCREMENT)
            if (self.camera.zoom < self.ZOOM_INCREMENT): self.camera.zoom = self.ZOOM_INCREMENT
        
        if pyray.is_key_down(87):
            self.pose_delta[0] = self.agent_vel * np.cos(np.pi*(self.agent_pose[2]/180))
            self.pose_delta[1] = self.agent_vel * np.sin(np.pi*(self.agent_pose[2]/180))
        if pyray.is_key_down(65):
            self.pose_delta[2] = self.agent_ang_vel
        if pyray.is_key_down(68):
            self.pose_delta[2] = -self.agent_ang_vel
        if pyray.is_key_down(83):
            self.pose_delta[0] = -self.agent_vel * np.cos(np.pi*(self.agent_pose[2]/180))
            self.pose_delta[1] = -self.agent_vel * np.sin(np.pi*(self.agent_pose[2]/180))
        if pyray.is_key_pressed(69):
            self.agent_vel += 1
            self.agent_vel = min(20, self.agent_vel)
            print("Velocity changed to {} pixels/s".format(self.agent_vel))
        if pyray.is_key_pressed(81):
            self.agent_vel -= 1
            self.agent_vel = max(1, self.agent_vel)
            print("Velocity changed to {} pixels/s".format(self.agent_vel))

        self.agent_pose += self.pose_delta
        if self.agent_pose[2] > 180:
            self.agent_pose[2] -= 360
        elif self.agent_pose[2] < -180:
            self.agent_pose[2] += 360
        self.pose_delta = np.zeros_like(self.pose_delta)

        if not self.read_event.is_set():
            self.write_event.set()
            self.update_agent_posegraph()
            self.write_event.clear()

    def update_agent_posegraph(self):
        dist_travel_xy = self.agent_pose[0:2] - self.agent_pose_history[-1][0:2]
        dist_travel_norm = np.linalg.norm(dist_travel_xy)
        if dist_travel_norm > 20:
            trans = dist_travel_norm + self.prng.normal(0., self.meas_noise_trans)
            rot1 = np.arctan2(dist_travel_xy[1], dist_travel_xy[0])*180/np.pi - self.agent_pose_history[-1][2]
            rot2 = self.agent_pose[2] - self.agent_pose_history[-1][2] - rot1
            rot1 += self.prng.normal(0.0, self.meas_noise_rot)
            rot2 += self.prng.normal(0.0, self.meas_noise_rot)

            x_noisy = trans*np.cos((self.var_theta_noisy[-1] + rot1)*np.pi/180)
            y_noisy = trans*np.sin((self.var_theta_noisy[-1] + rot1)*np.pi/180)
            # x_noisy = trans*np.cos((self.agent_pose_history[-1][2] + rot1)*np.pi/180)  # Uses GT yaw
            # y_noisy = trans*np.sin((self.agent_pose_history[-1][2] + rot1)*np.pi/180)  # Uses GT yaw
            
            new_var_node = VariableNode(self.n_vars, 2)
            new_var_node.prior.eta = (self.last_pose_var.mu + np.array([x_noisy, y_noisy])) * self.lambda_prior 
            new_var_node.prior.lam = np.diag([self.lambda_prior, self.lambda_prior])
            new_var_node.belief.eta = new_var_node.prior.eta.copy()
            new_var_node.belief.lam = new_var_node.prior.lam.copy() 
            new_var_node.Sigma = 1/np.diagonal(new_var_node.belief.lam)
            new_var_node.mu = new_var_node.Sigma * new_var_node.belief.eta
            new_var_node.type = "pose"
            new_var_node.GT = self.agent_pose[0:2].copy()

            self.var_nodes.append(new_var_node)

            new_factor = Factor(self.n_factors,
                                [self.last_pose_var, self.var_nodes[-1]],
                                [x_noisy, y_noisy],
                                self.meas_noise,
                                linear_displacement.meas_fn,
                                linear_displacement.jac_fn,
                                loss=None,
                                mahalanobis_threshold=2,
                                wildfire=self.b_wild)
            
            new_factor.type = "pose - pose"
                        
            self.factors.append(new_factor)

            self.var_theta_noisy.append(self.var_theta_noisy[-1] + rot1 + rot2)
            self.agent_pose_history.append(self.agent_pose.copy())

            self.last_pose_var = self.var_nodes[-1]

            self.n_vars += 1
            self.n_factors += 1

            for l_id in range(len(self.landmark_pos)):
                dist_agent2landmark_xy = self.landmark_pos[l_id] - self.agent_pose[0:2]
                if np.linalg.norm(dist_agent2landmark_xy) < self.agent_radius:
                    dist_agent2landmark_ang = self.agent_pose[2] - 180/np.pi * np.arctan2(dist_agent2landmark_xy[1], dist_agent2landmark_xy[0])
                    if np.abs((dist_agent2landmark_ang+180) % 360 - 180) < self.agent_angle/2:
                        if self.landmark_vars[l_id]:
                            l_var = self.landmark_vars[l_id]
                            meas = dist_agent2landmark_xy + self.prng.normal(0., self.meas_noise, 2)

                            new_factor = Factor(self.n_factors,
                                [self.last_pose_var, l_var],
                                meas,
                                self.meas_noise,
                                linear_displacement.meas_fn,
                                linear_displacement.jac_fn,
                                loss=None,
                                mahalanobis_threshold=2)
                            
                            new_factor.type = "landmark - pose"
            
                            self.factors.append(new_factor)

                            self.n_factors += 1
                        else:
                            meas = dist_agent2landmark_xy + self.prng.normal(0., self.meas_noise, 2)

                            new_var_node = VariableNode(self.n_vars, 2)
                            new_var_node.prior.eta = (self.last_pose_var.mu + meas) * self.lambda_prior
                            new_var_node.prior.lam = np.diag([self.lambda_prior, self.lambda_prior])
                            new_var_node.belief.eta = new_var_node.prior.eta.copy()
                            new_var_node.belief.lam = new_var_node.prior.lam.copy()
                            new_var_node.Sigma = 1/np.diagonal(new_var_node.belief.lam)
                            new_var_node.mu = new_var_node.Sigma * new_var_node.belief.eta
                            new_var_node.type = "landmark"
                            new_var_node.GT = self.landmark_pos[l_id]

                            self.var_nodes.append(new_var_node)
                            self.landmark_vars[l_id] = self.var_nodes[-1]

                            new_factor = Factor(self.n_factors,
                                [self.last_pose_var, self.var_nodes[-1]],
                                meas,
                                self.meas_noise,
                                linear_displacement.meas_fn,
                                linear_displacement.jac_fn,
                                loss=None,
                                mahalanobis_threshold=2)
                            
                            new_factor.type = "landmark - pose"
            
                            self.factors.append(new_factor)

                            self.n_vars += 1
                            self.n_factors += 1


    def draw_nodes(self):
        if not self.b_show_active:
            if self.show_layer == 0:        
                for factor in self.factors:
                    if factor.type[0:5] != "multi" and factor.type != "dead":
                        if factor.type == "landmark - pose":
                            line_colour = pyray.BLUE
                        else:
                            line_colour = pyray.GREEN

                        pyray.draw_line_ex(pyray.Vector2(factor.adj_var_nodes[0].mu[0], self.SCREEN_HEIGHT - factor.adj_var_nodes[0].mu[1]),
                                        pyray.Vector2(factor.adj_var_nodes[1].mu[0], self.SCREEN_HEIGHT - factor.adj_var_nodes[1].mu[1]), 1, line_colour)
                    
                if not self.b_multi:
                    for var in self.var_nodes:
                        if var.type == "landmark":
                            circle_colour = pyray.PURPLE
                        else:
                            circle_colour = pyray.RED

                        pyray.draw_circle(int(var.mu[0]), self.SCREEN_HEIGHT - int(var.mu[1]), self.node_size, circle_colour)
                        pyray.draw_text(str(var.variableID), int(var.mu[0]), self.SCREEN_HEIGHT - int(var.mu[1]), int(self.node_size/2), pyray.WHITE)
                        if all(var.residual > 1):
                            pos2 = var.mu + var.residual
                            pyray.draw_line_ex(pyray.Vector2(var.mu[0], self.SCREEN_HEIGHT - var.mu[1]), pyray.Vector2(pos2[0], self.SCREEN_HEIGHT - pos2[1]), 2, pyray.PINK)

                else:
                    for var in self.var_nodes:
                        if var.multigrid.level == 0:
                            if var.multigrid.parent:
                                parent = var.multigrid.parent
                                while parent.multigrid.parent:
                                    parent = parent.multigrid.parent
                                if parent.multigrid.level == 1:
                                    circle_colour = pyray.ORANGE
                                elif parent.multigrid.level == 2:
                                    circle_colour = pyray.YELLOW
                                else:
                                    circle_colour = pyray.WHITE
                                id_text = str(parent.variableID)

                            else:
                                if var.type == "landmark":
                                    circle_colour = pyray.PURPLE
                                else:
                                    circle_colour = pyray.RED 
                                id_text = str(var.variableID)

                            pyray.draw_circle(int(var.mu[0]), self.SCREEN_HEIGHT - int(var.mu[1]), self.node_size, circle_colour)
                            pyray.draw_text(id_text, int(var.mu[0]), self.SCREEN_HEIGHT - int(var.mu[1]), int(self.node_size/2), pyray.WHITE)
                            if all(var.residual > 1):
                                pos2 = var.mu + var.residual
                                pyray.draw_line_ex(pyray.Vector2(var.mu[0], self.SCREEN_HEIGHT - var.mu[1]), pyray.Vector2(pos2[0], self.SCREEN_HEIGHT - pos2[1]), 2, pyray.PINK)
                                    
            elif self.b_multi:
                if self.show_layer == 1:
                    colour = pyray.ORANGE
                elif self.show_layer == 2:
                    colour = pyray.YELLOW
                else:
                    colour = pyray.WHITE

                line_colour = pyray.GREEN

                try:   
                    for factor in self.graph.multigrid_factors[self.show_layer]:
                        pos=[]
                        for var in factor.adj_var_nodes:
                            child = var.multigrid.child
                            while child.multigrid.child:
                                child = child.multigrid.child
                            pos.append([child.mu[0],child.mu[1]])
                        
                        pyray.draw_line_ex(pyray.Vector2(pos[0][0], self.SCREEN_HEIGHT - pos[0][1]),
                                pyray.Vector2(pos[1][0], self.SCREEN_HEIGHT - pos[1][1]), 1, line_colour)

                        
                    for var in self.graph.multigrid_vars[self.show_layer]:
                        child = var.multigrid.child
                        while child.multigrid.child:
                            child = child.multigrid.child

                        pyray.draw_circle(int(child.mu[0]), self.SCREEN_HEIGHT - int(child.mu[1]), self.node_size, colour)
                        pyray.draw_text(str(var.variableID), int(child.mu[0]), self.SCREEN_HEIGHT - int(child.mu[1]), int(self.node_size/2), pyray.WHITE)

                        pos2 = np.array([int(child.mu[0]), int(child.mu[1])]) + var.mu
                        pyray.draw_line_ex(pyray.Vector2(int(child.mu[0]), self.SCREEN_HEIGHT - int(child.mu[1])), pyray.Vector2(pos2[0], self.SCREEN_HEIGHT - pos2[1]), 2, pyray.BLUE)

                except:
                    pyray.draw_text("NO VARIABLES ON THIS LAYER", 50,
                    int(self.SCREEN_HEIGHT/2), 50, pyray.WHITE)

            else:
                pyray.draw_text("MULTIGRID NOT SELECTED", 50,
                                int(self.SCREEN_HEIGHT/2), 100, pyray.WHITE)
            
        else:
            line_colour = pyray.GREEN
            for factor in self.graph.multigrid_factors[0]:
                if factor.active:
                    pyray.draw_line_ex(pyray.Vector2(factor.adj_var_nodes[0].mu[0], self.SCREEN_HEIGHT - factor.adj_var_nodes[0].mu[1]),
                                pyray.Vector2(factor.adj_var_nodes[1].mu[0], self.SCREEN_HEIGHT - factor.adj_var_nodes[1].mu[1]), 1, line_colour)

            for var in self.graph.multigrid_vars[0]:
                circle_colour = pyray.RED
                if var.active:
                        pyray.draw_circle(int(var.mu[0]), self.SCREEN_HEIGHT - int(var.mu[1]), self.node_size, circle_colour)
                        pyray.draw_text(str(var.variableID), int(var.mu[0]), self.SCREEN_HEIGHT - int(var.mu[1]), int(self.node_size/2), pyray.WHITE)
                        if all(var.residual > 1):
                            pos2 = var.mu + var.residual
                            pyray.draw_line_ex(pyray.Vector2(var.mu[0], self.SCREEN_HEIGHT - var.mu[1]), pyray.Vector2(pos2[0], self.SCREEN_HEIGHT - pos2[1]), 2, pyray.PINK)
                else:
                    if var.multigrid.parent:
                        parent = var.multigrid.parent
                        while not parent.active:
                            parent = parent.multigrid.parent
                        level = parent.multigrid.level
                        if level == 0:
                            circle_colour = pyray.RED
                        elif level == 1:
                            circle_colour = pyray.YELLOW
                        elif level == 2:
                            circle_colour = pyray.ORANGE
                        else:
                            circle_colour = pyray.WHITE
                        line_colour = circle_colour
                        pyray.draw_circle(int(var.mu[0]), self.SCREEN_HEIGHT - int(var.mu[1]), self.node_size, circle_colour)
                        for factor in parent.adj_factors:
                            for adj_var in factor.adj_var_nodes:
                                if adj_var != parent:
                                    child = adj_var.multigrid.child
                                    while child.multigrid.child:
                                        child = adj_var.multigrid.child
                                    pyray.draw_circle(int(child.mu[0]), self.SCREEN_HEIGHT - int(child.mu[1]), self.node_size, circle_colour)
                            pyray.draw_line_ex(pyray.Vector2(var.mu[0], self.SCREEN_HEIGHT - var.mu[1]), pyray.Vector2(child.mu[0], self.SCREEN_HEIGHT - child.mu[1]), 2, line_colour)
        

    def add_landmark(self, x, y):

        y = self.SCREEN_HEIGHT - y

        self.landmark_pos.append([x,y])
        self.landmark_vars.append(None)


    def generate_random_landmarks(self):

        for _ in range(self.n_rand):
            x = int(self.prng.uniform(-self.SCREEN_WIDTH/2, self.SCREEN_WIDTH*1.5))
            y = int(self.prng.uniform(-self.SCREEN_HEIGHT/2, self.SCREEN_HEIGHT*1.5))
            self.add_landmark(x, y)

    def clear_nodes(self):
        self.landmark_vars = [None for _ in self.landmark_pos]
        self.landmark_pos = []
        self.show_layer = 0
        self.b_GBP_scenario = False
        self.read_event.clear()

    def reset(self):
        self.b_run = False
        self.reset_event.set()
        self.n_vars = 0
        self.n_factors = 0
        self.agent_pose_history = [self.agent_pose.copy()]
        self.var_nodes = []
        self.factors = []
        self.landmark_vars = [None for _ in self.landmark_pos]
        self.show_layer = 0
        self.b_GBP_scenario = False

        new_var_node = VariableNode(self.n_vars, 2)
        new_var_node.prior.eta = (self.agent_pose[0:2].copy()) * self.lambda_anchor
        new_var_node.prior.lam = np.diag([self.lambda_anchor, self.lambda_anchor])
        new_var_node.belief.eta = new_var_node.prior.eta.copy()
        new_var_node.belief.lam = new_var_node.prior.lam.copy()
        new_var_node.Sigma = 1/np.diagonal(new_var_node.belief.lam)
        new_var_node.mu = new_var_node.Sigma * new_var_node.belief.eta
        new_var_node.type = "pose"
        new_var_node.GT = self.agent_pose[0:2].copy()

        self.var_nodes.append(new_var_node)
        self.n_vars += 1
    
        self.var_theta_noisy = [self.agent_pose[2]]

        self.last_pose_var = self.var_nodes[-1]

    def save_data(self):
        os.chdir('/home/callum/gbp-private-master')
        cwd = os.getcwd()
        with open(cwd + '/data/2d/gt_measurements.txt','w') as f:
            for item in self.measGT_2D:
                f.writelines(str(item)+'\n')

        with open(cwd + '/data/2d/noisy_measurements.txt','w') as f:
            for item in self.measNoisy_2D:
                f.writelines(str(item) + '\n')
        
        with open(cwd + '/data/2d/meas_variances.txt','w') as f:
            for item in self.measGT_2D:
                f.writelines(str(self.meas_noise) + '\n')
        
        with open(cwd + '/data/2d/factor_potentials_eta.txt','w') as f:
            for edge in range(int(len(self.measGT_2D)/2)):
                f.writelines(str(self.measGT_2D[edge*self.dof]/self.meas_noise) + '\n')
                f.writelines(str(self.measGT_2D[edge*self.dof+1]/self.meas_noise) + '\n')     
                f.writelines(str(-self.measGT_2D[edge*self.dof]/self.meas_noise) + '\n')
                f.writelines(str(-self.measGT_2D[edge*self.dof+1]/self.meas_noise) + '\n')       

        with open(cwd + '/data/2d/factor_potentials_lambda.txt','w') as f:
            for edge in range(int(len(self.measGT_2D)/2)):
                f.writelines(str(1/self.meas_noise) + '\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(1/self.meas_noise) + '\n')
                f.writelines(str(-1/self.meas_noise) + '\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(-1/self.meas_noise) + '\n')
                f.writelines(str(-1/self.meas_noise) + '\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(-1/self.meas_noise) + '\n')
                f.writelines(str(1/self.meas_noise) + '\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(1/self.meas_noise) + '\n')

        with open(cwd + '/data/2d/measurements_nodeIDs.txt','w') as f:
            #self.connections.sort()
            for item in self.connections:
                f.writelines(str(item[0])+'\n')
                f.writelines(str(item[1])+'\n')
        
        with open(cwd + '/data/2d/n_edges.txt','w') as f:
            f.writelines(str(int(len(self.measGT_2D)/2))+'\n')

        with open(cwd + '/data/2d/var_nodes_dofs.txt','w') as f:
            for i in range(self.n_vars):
                f.writelines(str(2)+'\n')

        with open(cwd + '/data/2d/var_dofs.txt','w') as f:
            f.writelines(str(2)+'\n')

        with open(cwd + '/data/2d/n_varnodes.txt','w') as f:
            f.writelines(str(self.n_vars)+'\n')

        with open(cwd + '/data/2d/priors_lambda.txt','w') as f:
            for i in range(self.n_vars):
                f.writelines(str(self.lambda_prior)+'\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(0)+'\n')
                f.writelines(str(self.lambda_prior)+'\n')

        with open(cwd + '/data/2d/priors_eta.txt','w') as f:
            for i in range(self.n_vars):
                # f.writelines(str(self.var_x[i]/self.scale * self.lambda_prior)+'\n')
                # f.writelines(str(self.var_y[i]/self.scale * self.lambda_prior)+'\n')
                f.writelines(str((0) * self.lambda_prior)+'\n')
                f.writelines(str((0) * self.lambda_prior)+'\n')

        with open(cwd + '/data/2d/num_edges_pernode.txt','w') as f:
            for i in range(self.n_vars):
                n_edges = 0
                for link in self.connections:
                    if any([k == i for k in link ]):
                        n_edges += 1

                f.writelines(str(n_edges)+'\n')
        
        print("---- DATA SAVED ----")