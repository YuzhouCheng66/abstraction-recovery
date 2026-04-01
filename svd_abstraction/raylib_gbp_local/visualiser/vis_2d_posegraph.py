import pyray
import numpy as np
from numpy.random import RandomState
import os
import threading
import time
from gbp.gbp import Factor
from gbp.gbp import VariableNode
from gbp.factors import linear_displacement
from gbp.gbp import NdimGaussian

class game:
    def __init__(self, graph=None, n_rand=100):
        self.prng = RandomState(69)
        # OPTIONS --------------------
        self.click_to_add = True
        self.b_run = False
        self.b_wild = False
        self.b_GBP_scenario = False
        self.b_pyamg = False
        self.b_show_plots = True
        self.IPU_process_running = False
        self.read_event = threading.Event()
        self.write_event = threading.Event()
        self.reset_event = threading.Event()
        self.pause_event = threading.Event()
        self.skip_event = threading.Event()
        self.stop_event = threading.Event()
        if graph is not None:
            self.graph = graph
            self.b_multi = graph.multigrid
        
        self.C = []
        self.C_var_ids = []
        self.C_base_ids = []
        self.layer_factor_ids = []
        self.show_layer = 0
        
        self.node_size = 10
        self.txt_off = int(self.node_size/2)
        self.scale = 1
        self.SCREEN_WIDTH = 1800
        self.SCREEN_HEIGHT = 900

        self.camera = pyray.Camera2D()
        self.camera.zoom = 1.0
        self.ZOOM_INCREMENT = 0.125

        self.means = []
        self.connections = []
        self.n_vars = 0
        self.n_factors = 0
        self.n_rand = n_rand
        self.n_rand_value = pyray.ffi.new("int *", int(n_rand))
        self.factors = []
        self.var_nodes = []

        self.meas_noise = 10.
        self.prior_noise = 100.
        self.prior_start_condition = "point"
        self.lambda_prior = 1e-5
        self.use_radius = False
        self.radius_connect = 100 # pixels
        self.k_neighbours = 4

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
        self.r_input = pyray.Rectangle(self.SCREEN_WIDTH - 150*2, 70*2, 30*2, 30*2)
        
        self.r_play = pyray.Rectangle(10, self.SCREEN_HEIGHT- 60, 50, 50)
        self.r_pause = pyray.Rectangle(60, self.SCREEN_HEIGHT- 60, 50, 50)
        self.r_skip = pyray.Rectangle(120, self.SCREEN_HEIGHT- 60, 50, 50)

        self.r_layer0 = pyray.Rectangle(420, self.SCREEN_HEIGHT- 60, 90, 50)
        self.r_layer1 = pyray.Rectangle(520, self.SCREEN_HEIGHT- 60, 90, 50)
        self.r_layer2 = pyray.Rectangle(620, self.SCREEN_HEIGHT- 60, 90, 50)
        self.r_layer3 = pyray.Rectangle(720, self.SCREEN_HEIGHT- 60, 90, 50)
        self.r_layer4 = pyray.Rectangle(820, self.SCREEN_HEIGHT- 60, 90, 50)

        self.c_true = pyray.Color(0,255,0,50)
        self.c_false = pyray.Color(255,0,0,50)
        self.c_orange = pyray.Color(255,165,0,50)


    def run(self):
        resource_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")

        self.stop_event.clear()
        pyray.init_window(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, "2D Posegraph parallel GBP")
        pyray.set_target_fps(30)
        pyray.gui_set_style(0,16,18)

        self.t_play = pyray.load_texture(os.path.join(resource_dir, "play_5050.png"))
        self.t_skip = pyray.load_texture(os.path.join(resource_dir, "skip_5050.png"))
        self.t_pause = pyray.load_texture(os.path.join(resource_dir, "pause_5050.png"))

        try:
            while not pyray.window_should_close():
                # update
                self.user_input()
                # draw
                pyray.begin_drawing()
                pyray.clear_background(pyray.BLACK)
                pyray.begin_mode_2d(self.camera)

                self.draw_nodes()

                pyray.end_mode_2d()

                self.draw_interfaces()
                self.draw_textures()

                pyray.end_drawing()
        finally:
            pyray.unload_texture(self.t_play)
            pyray.unload_texture(self.t_skip)
            pyray.unload_texture(self.t_pause)
            pyray.close_window()
            self.b_run = False
            self.reset_event.set()
            self.stop_event.set()


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
            pyray.gui_value_box(self.r_input, "", self.n_rand_value, 0, 10000, False)
            self.n_rand = int(self.n_rand_value[0])

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
                    self.generate_random_nodes()
                # elif pyray.check_collision_point_rec(mouseXY, self.r_input):
                #     done = False
                #     input = ""
                #     while not done:
                #         key = pyray.get_key_pressed()
                #         print(key)
                #         if 48 < key < 57:
                #             input += str(key)
                #         elif key == 89:
                #             done = True
                #         elif key != 0:
                #             input[-1] = ""
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
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer1):
                    self.show_layer = 1
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer2):
                    self.show_layer = 2
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer3):
                    self.show_layer = 3
                elif pyray.check_collision_point_rec(mouseXY, self.r_layer4):
                    self.show_layer = 4
                elif self.click_to_add and not self.b_run:
                    mouseWorldPos = pyray.get_screen_to_world_2d(pyray.get_mouse_position(), self.camera)
                    self.add_node_near(mouseWorldPos.x, mouseWorldPos.y)

                self.node_size = 15 - min(5 * (self.n_vars/100),10)
                self.txt_off = int(self.node_size/2)

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


    def draw_nodes(self):
        if self.show_layer == 0:        
            for factor in self.factors:
                if factor.type[0:5] != "multi" and factor.type != "dead":
                    line_colour = pyray.GREEN

                    pyray.draw_line_ex(pyray.Vector2(factor.adj_var_nodes[0].mu[0], self.SCREEN_HEIGHT - factor.adj_var_nodes[0].mu[1]),
                                    pyray.Vector2(factor.adj_var_nodes[1].mu[0], self.SCREEN_HEIGHT - factor.adj_var_nodes[1].mu[1]), 1, line_colour)
                
            if not self.b_multi:
                for var in self.var_nodes:
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

    
    def add_node_near(self, x, y):

        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]

        for id, val in enumerate(y):
            y[id] = self.SCREEN_HEIGHT - val

        start_id = self.n_vars

        for id in range(len(x)):
            match self.prior_start_condition:
                case "random":
                    x_prior = np.random.uniform(0, self.SCREEN_WIDTH/1.1)
                    y_prior = np.random.uniform(0, self.SCREEN_HEIGHT/1.1)
                    prior_pos = np.array([x_prior, y_prior])
                case "near":
                    prior_pos = np.array([x[id],y[id]]) + self.prng.normal(0., self.prior_noise,2)
                case "point":
                    prior_pos = np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])
            new_var_node = VariableNode(self.n_vars, 2)
            new_var_node.prior.eta = prior_pos * self.lambda_prior
            new_var_node.prior.lam = np.diag([self.lambda_prior, self.lambda_prior])
            new_var_node.belief.eta = new_var_node.prior.eta.copy()
            new_var_node.belief.lam = new_var_node.prior.lam.copy() 
            new_var_node.Sigma = 1/np.diagonal(new_var_node.belief.lam)
            new_var_node.mu = new_var_node.Sigma * new_var_node.belief.eta
            new_var_node.type = "pose"
            new_var_node.GT = np.array([x[id],y[id]])

            self.var_nodes.append(new_var_node)
            self.connections.append([])
            self.n_vars += 1

        end_id = self.n_vars

        for id0 in range(start_id,end_id):
            for id1 in range(self.n_vars):
                if id1 != id0:
                    dist = np.linalg.norm(self.var_nodes[id1].GT - self.var_nodes[id0].GT)
                    if dist <= self.radius_connect:
                        meas = self.var_nodes[id1].GT - self.var_nodes[id0].GT + self.prng.normal(0., self.meas_noise,2)

                        new_factor = Factor(self.n_factors,
                                    [self.var_nodes[id0], self.var_nodes[id1]],
                                    meas,
                                    self.meas_noise,
                                    linear_displacement.meas_fn,
                                    linear_displacement.jac_fn,
                                    loss=None,
                                    mahalanobis_threshold=2)
                        
                        new_factor.type = "pose - pose"
                        
                        self.factors.append(new_factor)
                        self.n_factors += 1

    def add_node_neighbour(self, x, y):

        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
       
        for id, val in enumerate(y):
            y[id] = self.SCREEN_HEIGHT - val

        start_id = self.n_vars

        for id in range(len(x)):
            match self.prior_start_condition:
                case "random":
                    x_prior = np.random.uniform(0, self.SCREEN_WIDTH/1.1)
                    y_prior = np.random.uniform(0, self.SCREEN_HEIGHT/1.1)
                    prior_pos = np.array([x_prior, y_prior])
                case "near":
                    prior_pos = np.array([x[id],y[id]]) + self.prng.normal(0., self.prior_noise,2)
                case "point":
                    prior_pos = np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])
            new_var_node = VariableNode(self.n_vars, 2)
            new_var_node.prior.eta = prior_pos * self.lambda_prior
            new_var_node.prior.lam = np.diag([self.lambda_prior, self.lambda_prior])
            new_var_node.belief.eta = new_var_node.prior.eta.copy()
            new_var_node.belief.lam = new_var_node.prior.lam.copy() 
            new_var_node.Sigma = 1/np.diagonal(new_var_node.belief.lam)
            new_var_node.mu = new_var_node.Sigma * new_var_node.belief.eta
            new_var_node.type = "pose"
            new_var_node.GT = np.array([x[id],y[id]])

            self.var_nodes.append(new_var_node)
            self.connections.append([])
            self.n_vars += 1

        end_id = self.n_vars

        for id0 in range(start_id,end_id):
            if len(self.connections[id0]) < self.k_neighbours: 
                dist_tmp = np.ones(self.n_vars) * np.infty
                for id1 in range(self.n_vars):
                    if id1 != id0 and id1 not in self.connections[id0]:
                        dist_tmp[id1]=np.linalg.norm(self.var_nodes[id1].GT - self.var_nodes[id0].GT)

                while len(self.connections[id0]) < self.k_neighbours:
                    id1 = np.argmin(dist_tmp)
                    meas = self.var_nodes[id1].GT - self.var_nodes[id0].GT

                    new_factor = Factor(self.n_factors,
                                [self.var_nodes[id0], self.var_nodes[id1]],
                                meas,
                                self.meas_noise,
                                linear_displacement.meas_fn,
                                linear_displacement.jac_fn,
                                loss=None,
                                mahalanobis_threshold=2)
                        
                    new_factor.type = "pose - pose"
                    self.factors.append(new_factor)

                    self.connections[id0].append(id1)

                    self.n_factors += 1

                    dist_tmp[id1] = np.infty


    def generate_random_nodes(self):
        x = []
        y = []

        for _ in range(self.n_rand):
            x.append(self.prng.uniform(0, self.SCREEN_WIDTH/1.5))
            y.append(self.prng.uniform(0, self.SCREEN_HEIGHT/1.5))
        
        if self.use_radius:
            self.add_nodes_near(x,y)
        else:
            self.add_node_neighbour(x,y)

    def clear_nodes(self):
        self.n_vars = 0
        self.n_factors = 0
        self.connections = []
        self.var_nodes = []
        self.factors = []
        self.show_layer = 0
        self.b_GBP_scenario = False

    def reset(self):
        self.b_run = False
        self.reset_event.set()
        while self.reset_event.isSet():
            time.sleep(0.1)
        vars_to_remove = []
        for var in self.var_nodes:
            if var.type[0:5] == "multi":
                vars_to_remove.append(var)
            else:
                var.adj_factors = []
                var.mu = var.prior.eta @ np.linalg.inv(var.prior.lam)
                var.Sigma = np.zeros([var.dofs, var.dofs])
                var.residual = np.zeros(var.dofs)
                var.belief.eta = var.prior.eta.copy()
                var.belief.lam = var.prior.lam.copy() 
        self.var_nodes = [var for var in self.var_nodes if var not in vars_to_remove]
        self.n_vars = len(self.var_nodes)
        factors_to_remove = []
        for factor in self.factors:
            if factor.type[0:5] == "multi":
                factors_to_remove.append(factor)
            else:
                factor.adj_beliefs = []
                factor.messages = []
                for adj_var_node in factor.adj_var_nodes:
                    factor.adj_beliefs.append(NdimGaussian(adj_var_node.dofs))
                    factor.messages.append(NdimGaussian(adj_var_node.dofs))#, eta=adj_var_node.prior.eta, lam=adj_var_node.prior.lam))
                    factor.factor = NdimGaussian(factor.dofs_conditional_vars)
                    factor.linpoint = np.zeros(factor.dofs_conditional_vars)
                    factor.residual = None
        self.factors = [factor for factor in self.factors if factor not in factors_to_remove]
        self.n_factors = len(self.factors)

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

    def generate_meaning_of_life(self):
        x = []
        y = []
        self.b_GBP_scenario = True

        with open('resources/meaning_of_life_x.txt') as f:
            lines = f.readlines() # list containing lines of file
            for line in lines:
                line = line.strip() # remove leading/trailing white spaces
                if line:
                   x.append(float(line))

        with open('resources/meaning_of_life_y.txt') as f:
            lines = f.readlines() # list containing lines of file
            for line in lines:
                line = line.strip() # remove leading/trailing white spaces
                if line:
                    y.append(self.SCREEN_HEIGHT - float(line))

        self.add_node_neighbour(x,y)

        # self.var_x = []
        # self.var_y = []

        # for id in range(self.n_vars):
        #     self.var_x.append(int(np.random.uniform(20, self.SCREEN_WIDTH/1.1)))
        #     self.var_y.append(int(np.random.uniform(20, self.SCREEN_HEIGHT/1.1)))
