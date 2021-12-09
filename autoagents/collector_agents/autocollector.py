import os
import math
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
import random
import string

from torch.distributions.categorical import Categorical

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs, _numpy

from autoagents.waypointer import Waypointer


#Added

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import is_within_distance_ahead, is_within_distance, compute_distance

def get_entry_point():
    return 'AutoCollector'

FPS = 20.
STOP_THRESH = 0.1
MAX_STOP = 500

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.1, theta=0.1, dt=0.1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class AutoCollector(AutonomousAgent):

    """
    action value agent but assumes a static world
    """

    def setup(self, path_to_conf_file):
        print('Collector agent setup')
        """
        Setup the agent parameters
        """

        self.track = Track.MAP
        self.num_frames = 0

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        device = torch.device('cuda')

        self.vizs = []
        self.rgbs = []
        self.segmentations = []
        self.lbls = []
        self.locs = [] # (x,y,z) absolute
        self.rots = [] # (pitch,yaw,roll) absolute
        self.spds = []
        self.cmds = []
        self.trafficlights = []
        self.cam_locations = [] # (x,y,z) absolute, differs from ego-vehicle location, offset between them always changes as expressed in world coordinates
        self.cam_rotations = [] # (pitch,yaw,roll) absolute, in our case same as ego-vehicle rotation as camera aligned

        self.waypointer = None

        if self.log_wandb:
            wandb.init(project='carla_data_phase1')

        self.noiser = OrnsteinUhlenbeckActionNoise(dt=1/FPS)
        self.prev_steer = 0

        self.stop_count = 0

        self.ego_cam = None

        ########################################################################
        # agents.navigation.BasicAgent default parameter
        ########################################################################
        target_speed = 5
        self._proximity_tlight_threshold = 5.0  # meters
        self._proximity_vehicle_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self.args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.4,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None

        self._local_planner = None
        self._vehicle = None
        self._world = None
        self._map = None
        ########################################################################

    def setup_environment(self):
        if self._world is None:
            self._world = CarlaDataProvider._client.get_world()
            vehicles = self._world.get_actors().filter('vehicle.*')
            for v in vehicles:
                if v.attributes['role_name'] == 'hero':
                    self._vehicle = v
                    break;
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed' : self._target_speed,
            'lateral_control_dict':self.args_lateral_dict})
        self._map = self._world.get_map()

        if self.ego_cam is None:
            cams = self._world.get_actors().filter('sensor.camera.rgb')
            for c in cams:
                print(c)
                if(c.attributes.get('fov') == '120'):
                    self.ego_cam = c
                    break;

    def destroy(self):
        if len(self.lbls) == 0:
            return

        self.flush_data()

    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })

        # Save data
        data_path = os.path.join(self.main_data_dir, _random_string())
        print ('Saving to {}'.format(data_path))

        lmdb_env = lmdb.open(data_path, map_size=int(1e10))

        length = len(self.lbls)
        with lmdb_env.begin(write=True) as txn:

            txn.put('len'.encode(), str(length).encode())

            for i in range(length):

                txn.put(
                    f'rgb_{i:04d}'.encode(),
                    np.ascontiguousarray(self.rgbs[i]).astype(np.uint8),
                )

                txn.put(
                    f'segmentation_{i:04d}'.encode(),
                    np.ascontiguousarray(self.segmentations[i]).astype(np.uint8),
                )

                txn.put(
                    f'birdview_{i:04d}'.encode(),
                    np.ascontiguousarray(self.lbls[i]).astype(np.uint8),
                )

                txn.put(
                    f'loc_{i:04d}'.encode(),
                    np.ascontiguousarray(self.locs[i]).astype(np.float32)
                )

                txn.put(
                    f'rot_{i:04d}'.encode(),
                    np.ascontiguousarray(self.rots[i]).astype(np.float32)
                )

                txn.put(
                    f'spd_{i:04d}'.encode(),
                    np.ascontiguousarray(self.spds[i]).astype(np.float32)
                )

                txn.put(
                    f'cmd_{i:04d}'.encode(),
                    np.ascontiguousarray(self.cmds[i]).astype(np.float32)
                )

                txn.put(
                    f'trafficlights_{i:04d}'.encode(),
                    np.ascontiguousarray(self.trafficlights[i]).astype(np.uint8),
                )

                txn.put(
                    f'cam_location_{i:04d}'.encode(),
                    np.ascontiguousarray(self.cam_locations[i]).astype(np.float32),
                )

                txn.put(
                    f'cam_rotation_{i:04d}'.encode(),
                    np.ascontiguousarray(self.cam_rotations[i]).astype(np.float32),
                )
        self.vizs.clear()
        self.rgbs.clear()
        self.segmentations.clear()
        self.lbls.clear()
        self.locs.clear()
        self.rots.clear()
        self.spds.clear()
        self.cmds.clear()
        self.trafficlights.clear()
        self.cam_locations.clear()
        self.cam_rotations.clear()

        lmdb_env.close()

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.map', 'id': 'MAP'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
        ]

        # Add sensors
        sensors.append({'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'width': 384, 'height': 160, 'fov': 120, 'id': 'rgb'})
        sensors.append({'type': 'sensor.camera.semantic_segmentation', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'width': 384, 'height': 160, 'fov': 120, 'id': 'segmentation'})
        return sensors

    def run_step(self, input_data, timestamp):
        debug=False
        rgbs = []
        segmentations = []

        _, rgb = input_data.get('rgb')
        _, segmentation = input_data.get('segmentation')

        rgbs.append(rgb[...,:3]) # Keeping R, G and B channels
        segmentations.append(segmentation[...,2]) # Labels are encoded in the 2nd channel


        _, lbl = input_data.get('MAP')
        _, col = input_data.get('COLLISION')
        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')

        if self._vehicle is None:
            self.setup_environment()

        cam_location = self.ego_cam.get_transform().location
        cam_location = np.array([cam_location.x, cam_location.y, cam_location.z])

        cam_rotation = self.ego_cam.get_transform().rotation
        cam_rotation = np.array([cam_rotation.pitch, cam_rotation.yaw, cam_rotation.roll])

        # Modify traffic light label of segmentation (discard it if no red light)
        relevant_traffic_light = CarlaDataProvider.get_next_traffic_light(self._vehicle, False)
        if relevant_traffic_light is not None:
            is_relevant_red_traffic_light_red = relevant_traffic_light.get_state() != carla.TrafficLightState.Green # Yellow is considered Red
        else:
            is_relevant_red_traffic_light_red = False
        # print(is_relevant_red_traffic_light_red)
        #print(1 in lbl[...,3])


        tls = (1 in lbl[...,3]) and is_relevant_red_traffic_light_red # Relevant traffic light is red and visible in the birdview
        #print(tls)
        #print('-------------------------------')
        if not tls:
            segmentation[segmentation == 18] = 0


        # if self.waypointer is None:
        #     self.waypointer = Waypointer(self._global_plan, gps)
        #     _, _, cmd = self.waypointer.tick(gps)
        # else:
        #     _, _, cmd = self.waypointer.tick(gps)

        yaw = ego.get('rot')[-1]
        rot = ego.get('rot') # roll, pitch, yaw
        rot = [rot[1], rot[2], rot[0]] # pitch, yaw, roll
        spd = ego.get('spd')
        loc = ego.get('loc')

        _, cmd = self._local_planner._waypoints_queue[0]
        print(cmd)


        # If it is idle, make it LANE_FOLLOW
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value


        ####################################################################################
        # Control and path planning is copied from agents.navigation.BasicAgent
        ####################################################################################

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step(debug=debug)

        throttle = control.throttle
        steer = control.steer
        brake = control.brake
        #########################################################################################

        if self.noise_collect:
            steer += self.noiser()

        flush_length = len(self.vizs)
        if flush_length > self.num_per_flush:
            self.flush_data()

        if (flush_length % 100) == 0:
            print(f'{flush_length}')

        spd = ego.get('spd')
        #self.vizs.append(visualize_obs(rgb, yaw/180*math.pi, (steer, throt, brake), spd, cmd=cmd.value, lbl=lbl_copy, sem=segmentation, tls=tls))
        self.vizs.append(np.zeros((1)))
        if col:
            self.flush_data()
            raise Exception('Collector has collided!! Heading out :P')

        if spd < STOP_THRESH:
            self.stop_count += 1
        else:
            self.stop_count = 0

        if cmd_value in [4,5]:
            actual_steer = steer
        else:
            actual_steer = steer * 1.2

        self.prev_steer = actual_steer

        # Save data
        if self.num_frames % (self.num_repeat+1) == 0 and self.stop_count < MAX_STOP:
            # Note: basically, this is like fast-forwarding. should be okay tho as spd is 0.
            self.rgbs.append(rgbs)
            self.segmentations.append(segmentations)
            self.lbls.append(lbl)
            self.locs.append(loc)
            self.rots.append(rot)
            self.spds.append(spd)
            self.cmds.append(cmd_value)
            self.trafficlights.append(tls)
            self.cam_locations.append(cam_location)
            self.cam_rotations.append(cam_rotation)

        self.num_frames += 1

        return carla.VehicleControl(steer=actual_steer, throttle=throttle, brake=brake)


    # agents.navigation.BasicAgent utilities
    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)

        self._local_planner.set_global_plan(route_trace)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def done(self):
        """
        Check whether the agent has reached its destination.
        :return bool
        """
        return self._local_planner.done()

    ################################################################################
    # agents.navigation.Agent utilities
    ################################################################################

    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(object_waypoint.transform,
                                        self._vehicle.get_transform(),
                                        self._proximity_tlight_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, traffic_light)

        return (False, None)

    def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    def _bh_is_vehicle_hazard(self, ego_wpt, ego_loc, vehicle_list,
                           proximity_th, up_angle_th, low_angle_th=0, lane_offset=0):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle. We also check the next waypoint, just to be
        sure there's not a sudden road id change.

        WARNING: This method is an approximation that could fail for very large
        vehicles, which center is actually on a different lane but their
        extension falls within the ego vehicle lane. Also, make sure to remove
        the ego vehicle from the list. Lane offset is set to +1 for right lanes
        and -1 for left lanes, but this has to be inverted if lane values are
        negative.

            :param ego_wpt: waypoint of ego-vehicle
            :param ego_log: location of ego-vehicle
            :param vehicle_list: list of potential obstacle to check
            :param proximity_th: threshold for the agent to be alerted of
            a possible collision
            :param up_angle_th: upper threshold for angle
            :param low_angle_th: lower threshold for angle
            :param lane_offset: for right and left lane changes
            :return: a tuple given by (bool_flag, vehicle, distance), where:
            - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
            - vehicle is the blocker object itself
            - distance is the meters separating the two vehicles
        """

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        for target_vehicle in vehicle_list:

            target_vehicle_loc = target_vehicle.get_location()
            # If the object is not in our next or current lane it's not an obstacle

            target_wpt = self._map.get_waypoint(target_vehicle_loc)
            if target_wpt.road_id != ego_wpt.road_id or \
                    target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=5)[0]
                if target_wpt.road_id != next_wpt.road_id or \
                        target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                    continue

            if is_within_distance(target_vehicle_loc, ego_loc,
                                  self._vehicle.get_transform().rotation.yaw,
                                  proximity_th, up_angle_th, low_angle_th):

                return (True, target_vehicle, compute_distance(target_vehicle_loc, ego_loc))

        return (False, None, -1)

    def _is_vehicle_hazard(self, vehicle_list):
        """

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_vehicle.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._proximity_vehicle_threshold):
                return (True, target_vehicle)

        return (False, None)


    @staticmethod
    def emergency_stop():
        """
        Send an emergency stop command to the vehicle

            :return: control for braking
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control


def _random_string(length=10):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))
