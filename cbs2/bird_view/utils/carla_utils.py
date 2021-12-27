import collections
import queue
import weakref
import time
import random

import numpy as np
import math

# import needed due to https://github.com/pytorch/pytorch/issues/36034
import torchvision
import carla

from carla import ColorConverter
from carla import WeatherParameters
from agents.tools.misc import draw_waypoints


from .map_utils import Wrapper as map_utils

PRESET_WEATHERS = {
    1: WeatherParameters.ClearNoon,
    2: WeatherParameters.CloudyNoon,
    3: WeatherParameters.WetNoon,
    4: WeatherParameters.WetCloudyNoon,
    5: WeatherParameters.MidRainyNoon,
    6: WeatherParameters.HardRainNoon,
    7: WeatherParameters.SoftRainNoon,
    8: WeatherParameters.ClearSunset,
    9: WeatherParameters.CloudySunset,
    10: WeatherParameters.WetSunset,
    11: WeatherParameters.WetCloudySunset,
    12: WeatherParameters.MidRainSunset,
    13: WeatherParameters.HardRainSunset,
    14: WeatherParameters.SoftRainSunset,
}

TRAIN_WEATHERS = {
    'clear_noon': WeatherParameters.ClearNoon,  # 1
    'wet_noon': WeatherParameters.WetNoon,  # 3
    'hardrain_noon': WeatherParameters.HardRainNoon,  # 6
    'clear_sunset': WeatherParameters.ClearSunset,  # 8
}

WEATHERS = list(TRAIN_WEATHERS.values())

BACKGROUND = [0, 47, 0] # Background (dark green)
COLORS = [
    (0, 179, 255), # Pedestrians (blue)
    (255, 255, 255), # Road lines (white)
    (70, 70, 70), # Road (dark grey)
    (250, 210, 1), # Vehicles (yellow)
    (204, 6, 5), # Red light (red)
]

TOWNS = ['Town01', 'Town02', 'Town03', 'Town04']
VEHICLE_NAME = 'vehicle.ford.mustang'


def dotproduct(v1, v2):
    return np.sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))

def angle_rowwise(A, B):
    p1 = np.einsum('ij,ij->i',A,B)
    p2 = np.linalg.norm(A,axis=1)
    p3 = np.linalg.norm(B,axis=1)
    p4 = p1 / (p2*p3)
    return np.degrees(np.arccos(np.clip(p4,-1.0,1.0)))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def is_within_distance_ahead(target_location, current_location, orientation,
                             max_distance, degree=60):
    u = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y])
    distance = np.linalg.norm(u)

    if distance > max_distance:
        return False

    v = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))])

    angle = math.degrees(math.acos(np.dot(u, v) / distance))

    return angle < degree


def carla_to_np_vec(data):
    return np.array([data.x, data.y, data.z])


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 0.1

    world.apply_settings(settings)


def carla_seg_to_np(carla_seg, colorConv=False):
    if colorConv:
        carla_seg.convert(ColorConverter.CityScapesPalette)

    img = np.frombuffer(carla_seg.raw_data, dtype=np.dtype('uint8'))
    img = np.reshape(img, (carla_seg.height, carla_seg.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]

    return img


def carla_img_to_np(carla_img):
    carla_img.convert(ColorConverter.Raw)

    img = np.frombuffer(carla_img.raw_data, dtype=np.dtype('uint8'))
    img = np.reshape(img, (carla_img.height, carla_img.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]

    return img


def get_birdview(observations):
    birdview = [
        observations['road'],
        observations['lane'],
        observations['traffic'],
        observations['vehicle'],
        observations['pedestrian']
    ]
    birdview = [x if x.ndim == 3 else x[..., None] for x in birdview]
    birdview = np.concatenate(birdview, 2)

    return birdview


def process(observations):
    result = dict()
    result['rgb'] = observations['rgb'].copy()
    result['birdview'] = observations['birdview'].copy()
    result['segmentation'] = observations['segmentation'].copy()
    result['traffic_lights'] = np.array(observations['traffic_lights']).copy()
    result['collided'] = observations['collided']
    result['is_at_red_light'] = observations['is_at_red_light']
    result['is_at_vehicle'] = observations['is_at_vehicle']
    result['is_at_walker'] = observations['is_at_walker']

    control = observations['control']
    control = [control.steer, control.throttle, control.brake]

    result['control'] = np.float32(control)

    measurements = [
        observations['position'],
        observations['orientation'],
        observations['velocity'],
        observations['position_cam'],
        observations['orientation_cam'],
        observations['acceleration'],
        observations['command'].value,
        observations['control'].steer,
        observations['control'].throttle,
        observations['control'].brake,
        observations['control'].manual_gear_shift,
        observations['control'].gear
    ]
    measurements = [x if isinstance(x, np.ndarray) else np.float32([x]) for x
                    in measurements]
    measurements = np.concatenate(measurements, 0)

    result['measurements'] = measurements

    return result


def visualize_birdview(birdview):
    """
    0 pedestrian
    1 road lines
    2 road
    3 vehicle
    4 red traffic light
    """
    h, w = birdview.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND

    for i in range(len(COLORS)):
        canvas[birdview[:, :, i] > 0] = COLORS[i]

    return canvas


def visualize_predicted_birdview(predicted, tau=0.5):
    # mask = np.concatenate([predicted.max(0)[np.newaxis]] * 7, 0)
    # predicted[predicted != mask] = 0
    # predicted[predicted == mask] = 1

    predicted[predicted < tau] = -1

    return visualize_birdview(predicted.transpose(1, 2, 0))


class PedestrianTracker(object):
    def __init__(self, wrapper, peds, ped_controllers, respawn_peds=True,
                 speed_threshold=0.1, stuck_limit=20):
        self._wrapper = wrapper()
        self._peds = peds
        self._ped_controllers = ped_controllers

        self._ped_timers = dict()
        for ped in peds:
            self._ped_timers[ped.id] = 0

        self._speed_threshold = speed_threshold
        self._stuck_limit = stuck_limit
        self._respawn_peds = respawn_peds

    def tick(self):
        for ped in self._peds:
            vel = ped.get_velocity()
            speed = np.linalg.norm([vel.x, vel.y, vel.z])

            if ped.id in self._ped_timers and speed < self._speed_threshold:
                self._ped_timers[ped.id] += 1
            else:
                self._ped_timers[ped.id] = 0

        stuck_ped_ids = []
        for ped_id, stuck_time in self._ped_timers.items():
            if stuck_time >= self._stuck_limit and self._respawn_peds:
                stuck_ped_ids.append(ped_id)

        ego_vehicle_location = self._wrapper._player.get_location()

        for ped_controller in self._ped_controllers:
            ped_id = ped_controller.parent.id
            if ped_id not in stuck_ped_ids:
                continue

            self._ped_timers.pop(ped_id)

            old_loc = ped.get_location()

            loc = None
            while True:
                _loc = self._wrapper._world.get_random_location_from_navigation()
                if _loc is not None and _loc.distance(
                        ego_vehicle_location) >= 10.0 \
                        and _loc.distance(old_loc) >= 10.0:
                    loc = _loc
                    break

            ped_controller.teleport_to_location(loc)
            # print("Teleported walker %d to %s" % (ped_id, loc))


class TrafficTracker(object):
    LANE_WIDTH = 5.0

    def __init__(self, agent, world):
        self._agent = agent
        self._world = world

        self._prev = None
        self._cur = None

        self.total_lights_ran = 0
        self.total_lights = 0
        self.ran_light = False

        self.last_light_id = -1

    def tick(self):
        self.ran_light = False
        self._prev = self._cur
        self._cur = self._agent.get_location()

        if self._prev is None or self._cur is None:
            return

        light = TrafficTracker.get_active_light(self._agent, self._world)
        active_light = light

        if light is not None and light.id != self.last_light_id:
            self.total_lights += 1
            self.last_light_id = light.id

        light = TrafficTracker.get_closest_light(self._agent, self._world)

        if light is None or light.state != carla.libcarla.TrafficLightState.Red:
            return

        light_location = light.get_transform().location
        light_orientation = light.get_transform().get_forward_vector()

        delta = self._cur - self._prev

        p = np.array([self._prev.x, self._prev.y])
        r = np.array([delta.x, delta.y])

        q = np.array([light_location.x, light_location.y])
        s = TrafficTracker.LANE_WIDTH * np.array(
            [-light_orientation.x, -light_orientation.y])

        if TrafficTracker.line_line_intersect(p, r, q, s):
            self.ran_light = True
            self.total_lights_ran += 1

    @staticmethod
    def get_closest_light(agent, world):
        location = agent.get_location()
        closest = None
        closest_distance = float('inf')

        for light in world.get_actors().filter('*traffic_light*'):
            delta = location - light.get_transform().location
            distance = np.sqrt(sum([delta.x ** 2, delta.y ** 2, delta.z ** 2]))

            if distance < closest_distance:
                closest = light
                closest_distance = distance

        return closest

    @staticmethod
    def get_active_light(ego_vehicle, world):

        _map = world.get_map()
        ego_vehicle_location = ego_vehicle.get_location()
        ego_vehicle_waypoint = _map.get_waypoint(ego_vehicle_location)

        lights_list = world.get_actors().filter('*traffic_light*')

        for traffic_light in lights_list:
            location = traffic_light.get_location()
            object_waypoint = _map.get_waypoint(location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue
            if object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if not is_within_distance_ahead(
                    location,
                    ego_vehicle_location,
                    ego_vehicle.get_transform().rotation.yaw,
                    10., degree=60):
                continue

            return traffic_light

        return None

    @staticmethod
    def line_line_intersect(p, r, q, s):
        r_cross_s = np.cross(r, s)
        q_minus_p = q - p

        if abs(r_cross_s) < 1e-3:
            return False

        t = np.cross(q_minus_p, s) / r_cross_s
        u = np.cross(q_minus_p, r) / r_cross_s

        if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
            return True

        return False


class CarlaWrapper(object):
    def __init__(
            self, town='Town01', vehicle_name=VEHICLE_NAME, port=2000,
            client=None,
            col_threshold=400, big_cam=False, seed=None, respawn_peds=True,
            **kwargs):

        if client is None:
            self._client = carla.Client('localhost', port)
        else:
            self._client = client

        self._client.set_timeout(100.0)

        set_sync_mode(self._client, False)

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()
        self._vehicle_bp = np.random.choice(
            self._blueprints.filter(vehicle_name))
        self._vehicle_bp.set_attribute('role_name', 'hero')

        self._tick = 0
        self._player = None

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)

        self._big_cam = big_cam
        self.col_threshold = col_threshold
        self.collided = False
        self._collided_frame_number = -1

        self.invaded = False
        self._invaded_frame_number = -1

        self.traffic_tracker = None

        self.n_vehicles = 0
        self.n_pedestrians = 0

        self._rgb_queue = None
        self.rgb_image = None
        self.seg_image = None
        self._seg_queue = None
        self._big_cam_queue = None
        self.big_cam_image = None

        self.seed = seed

        self._respawn_peds = respawn_peds
        self.disable_two_wheels = False

        self.tls_locs = None
        self.radius = None
        self.max_dist_traffic_light = None

    def spawn_vehicles(self):

        blueprints = self._blueprints.filter('vehicle.*')
        if self.disable_two_wheels:
            blueprints = [x for x in blueprints if
                          int(x.get_attribute('number_of_wheels')) == 4]
        spawn_points = self._map.get_spawn_points()

        for i in range(self.n_vehicles):
            blueprint = np.random.choice(blueprints)
            blueprint.set_attribute('role_name', 'autopilot')

            if blueprint.has_attribute('color'):
                color = np.random.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(
                    blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            vehicle = None
            while vehicle is None:
                vehicle = self._world.try_spawn_actor(blueprint,
                                                      np.random.choice(
                                                          spawn_points))

            vehicle.set_autopilot(True)
            vehicle.start_dtcrowd()

            self._actor_dict['vehicle'].append(vehicle)

        print("spawned %d vehicles" % len(self._actor_dict['vehicle']))

    def spawn_pedestrians(self, n_pedestrians):
        SpawnActor = carla.command.SpawnActor

        peds_spawned = 0

        walkers = []
        controllers = []

        while peds_spawned < n_pedestrians:
            spawn_points = []
            _walkers = []
            _controllers = []

            for i in range(n_pedestrians - peds_spawned):
                spawn_point = carla.Transform()
                loc = self._world.get_random_location_from_navigation()

                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

            blueprints = self._blueprints.filter('walker.pedestrian.*')
            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprints)

                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')

                batch.append(SpawnActor(walker_bp, spawn_point))

            for result in self._client.apply_batch_sync(batch, True):
                if result.error:
                    print(result.error)
                else:
                    peds_spawned += 1
                    _walkers.append(result.actor_id)

            walker_controller_bp = self._blueprints.find(
                'controller.ai.walker')
            batch = [
                SpawnActor(walker_controller_bp, carla.Transform(), walker) for
                walker in _walkers]

            for result in self._client.apply_batch_sync(batch, True):
                if result.error:
                    print(result.error)
                else:
                    _controllers.append(result.actor_id)

            controllers.extend(_controllers)
            walkers.extend(_walkers)

        print("spawned %d pedestrians" % len(controllers))

        return self._world.get_actors(walkers), self._world.get_actors(
            controllers)

    def set_weather(self, weather_string):
        if weather_string == 'random':
            weather = np.random.choice(WEATHERS)
        elif weather_string in TRAIN_WEATHERS:
            weather = TRAIN_WEATHERS[weather_string]
        else:
            weather = weather_string

        self.weather = weather
        self._world.set_weather(weather)

    def init(self, start=0, weather='random', n_vehicles=0, n_pedestrians=0,
             radius=10, dist=2):
        self.radius = radius
        self.max_dist_traffic_light = dist

        while True:
            self.n_vehicles = n_vehicles or self.n_vehicles
            self.n_pedestrians = n_pedestrians or self.n_pedestrians
            self._start_pose = self._map.get_spawn_points()[start]

            self.clean_up()
            self.spawn_player()
            self._setup_sensors()

            # Hiding away the gore.
            map_utils.init(self._client, self._world, self._map,
                           self._player, hero_cam=self.seg_camera)

            # Deterministic.
            if self.seed is not None:
                np.random.seed(self.seed)

            self.set_weather(weather)

            # Spawn vehicles
            self.spawn_vehicles()

            # Spawn pedestrians
            peds, ped_controllers = self.spawn_pedestrians(self.n_pedestrians)
            self._actor_dict['pedestrian'].extend(peds)
            self._actor_dict['ped_controller'].extend(ped_controllers)

            self.peds_tracker = PedestrianTracker(weakref.ref(self),
                                                  self.pedestrians,
                                                  self.ped_controllers,
                                                  respawn_peds=self._respawn_peds)

            self.traffic_tracker = TrafficTracker(self._player, self._world)
            self.init_tls()

            ready = self.ready()
            if ready:
                break

    def init_tls(self):
        # Get all traffic lights' states and trigger box locations
        self.tls = self.get_all_traffic_lights()
        # tls = self.filter_traffic_lights(tls)
        if len(self.tls) == 0:
            raise NotImplemented("No traffic lights found")

        self.tls_locs = np.array([carla_to_np_vec(i) for i, _ in self.tls])

    def spawn_player(self):
        self._player = self._world.spawn_actor(self._vehicle_bp,
                                               self._start_pose)
        self._player.set_autopilot(False)
        self._player.start_dtcrowd()
        self._actor_dict['player'].append(self._player)

    def ready(self, ticks=50):
        self.tick()
        self.get_observations()

        for controller in self._actor_dict['ped_controller']:
            controller.start()
            controller.go_to_location(
                self._world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())

        for _ in range(ticks):
            self.tick()
            self.get_observations()

        with self._rgb_queue.mutex:
            self._rgb_queue.queue.clear()

        with self._seg_queue.mutex:
            self._seg_queue.queue.clear()

        self._time_start = time.time()
        self._tick = 0

        print("Initial collided: %s" % self.collided)

        return not self.collided

    def tick(self):
        self._world.tick()
        self._tick += 1

        # More hiding.
        map_utils.tick()

        self.traffic_tracker.tick()
        self.peds_tracker.tick()

        # Put here for speed (get() busy polls queue).
        while self.rgb_image is None or self._rgb_queue.qsize() > 0:
            self.rgb_image = self._rgb_queue.get()

        while self.seg_image is None or self._seg_queue.qsize() > 0:
            self.seg_image = self._seg_queue.get()

        if self._big_cam:
            while self.big_cam_image is None or self._big_cam_queue.qsize() > 0:
                self.big_cam_image = self._big_cam_queue.get()

        return True

    def filter_traffic_lights(self, tls):
        """Filter given traffic lights based on euclidean distance.
        Returns: all traffic lights that fall in radius euclidean distance
        from world player."""

        def calc_angle(vec1, vec2):
            unit_vector_1 = vec1 / np.linalg.norm(vec1)
            unit_vector_2 = vec2 / np.linalg.norm(vec2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)

            return np.rad2deg(angle)

        new_tls = []
        player_tf = self._player.get_transform()
        fwd_vec_ego = carla_to_np_vec(player_tf.rotation.get_forward_vector())
        for tl_loc, tl_state in tls:
            # Calculate euclidean distance
            point_vec = carla_to_np_vec(player_tf.location) - carla_to_np_vec(
                tl_loc)
            dist = np.linalg.norm(point_vec)

            # Calculate angle in 2D space
            angle = calc_angle(fwd_vec_ego, point_vec)
            if dist <= self.radius and abs(angle) <= 90:
                new_tls.append([tl_loc, tl_state])

        return new_tls

    def get_all_traffic_lights(self):
        """Get all traffic light actors in the world. Get per traffic light
        its trigger box center location and its state.
        Returns: Nx2 array where N is number of traffic lights."""
        tls = []
        for tl in self._world.get_actors().filter('*traffic_light*'):
            # Trigger / bounding boxes contain half extent and offset relative to the actor.
            trigger_transform = tl.get_transform()
            trigger_transform.location += tl.trigger_volume.location
            # trigger_extent = tl.trigger_volume.extent
            # draw_waypoints(self._world, [trigger_transform], loc=True)
            tls.append([trigger_transform.location, tl.get_state()])

        return tls

    def check_trajectories(self, current_wp, dist=0.):
        # Check distance from current waypoint to every traffic light bbox.
        wp = carla_to_np_vec(current_wp.transform.location)
        dists = np.linalg.norm(self.tls_locs[:, :2] - wp[:2], axis=1)

        # Base case
        if dist >= self.radius or current_wp.is_junction:
            return

        if current_wp.is_junction:
            return 5

        mask_dists = dists <= self.max_dist_traffic_light
        fwd_2dvec = carla_to_np_vec(current_wp.transform.rotation.get_forward_vector())[:2]

        project_dists = np.abs(np.cross(fwd_2dvec, wp[:2] - self.tls_locs[:, :2])) / np.linalg.norm(fwd_2dvec)
        mask_projection = project_dists < (current_wp.lane_width / 2)

        mask = mask_dists & mask_projection
        if np.any(mask):
            # Success, return traffic light index
            # Multiple tls confirm distances and projection
            if np.sum(mask) > 1:
                print(dists[mask])
                print(np.min(dists[mask]))
                print(np.argwhere(dists == np.min(dists[mask])))
                raise NotImplemented("Oops, multiple tls")
                return np.argwhere(dists == np.min(dists[mask]))[0]

            return np.argmax(mask)
        else:
            # Failure, continue to next waypoint
            next_waypoints = list(current_wp.next(0.5))
            if len(next_waypoints) > 1:
                return None

            # Prevent infinite recursion
            next_wp = next_waypoints[0]
            delta_dist = np.linalg.norm(carla_to_np_vec(next_wp.transform.location) - wp)
            if delta_dist <= 0.1:
                return None

            return self.check_trajectories(next_wp, dist=dist+delta_dist)

    #
    # def set_radius(self, radius):
    #     self.radius = radius
    #
    # def set_max_dist_traffic_light(self, dist):
    #     self.max_dist_traffic_light = dist

    def get_traffic_lights_state(self):
        """Get the state of the relevant traffic light.
        Returns 0 for no traffic light, 1 for green, 2 for yellow, and 3 for red."""
        if self.radius is None or self.max_dist_traffic_light is None:
            raise ValueError("Radius ({}) or max dist traffic light ({}) not set. "
                             "Use set_radius() or set_max_dist_traffic_light().".format(self.radius, self.max_dist_traffic_light))

        # Check if multiple trajectories are available en route
        current_waypoint = self._world.get_map().get_waypoint(self._player.get_location())
        trajectory_tls = self.check_trajectories(current_waypoint, dist=0)

        if trajectory_tls is None:
            return 0

        if trajectory_tls == 5:
            return 5
        # selected = np.array(tls)[trajectory_tls]
        # object_waypoint = self._map.get_waypoint(carla.Location(selected[0]))

        # # Additional check
        # if object_waypoint.road_id != current_waypoint.road_id:
        #     return 4
        # if object_waypoint.lane_id != current_waypoint.lane_id:
        #     return 5
        # Return state of selected traffic light
        mapping = {'Green': 1, 'Yellow': 2, 'Red': 3}
        return mapping[str(np.array(self.tls)[trajectory_tls][1])]

    def get_observations(self):
        result = dict()
        result.update(map_utils.get_observations())


        # temp = self.seg_camera.get_transform()
        # # print("TESTING", temp)
        # cam_transform_loc = np.float32([temp.location.x,
        #                                 temp.location.y,
        #                                 temp.location.z])
        # cam_transform_rot = np.float32([temp.rotation.yaw,
        #                                 temp.rotation.roll,
        #                                 temp.rotation.pitch])
        # print("TESTING2", cam_transform_loc, cam_transform_rot)

        # print ("%.3f, %.3f"%(self.rgb_image.timestamp, self._world.get_snapshot().timestamp.elapsed_seconds))
        result.update({
            'rgb': carla_img_to_np(self.rgb_image),
            'birdview': get_birdview(result),
            'segmentation': carla_seg_to_np(self.seg_image, colorConv=False),
            'traffic_lights': self.get_traffic_lights_state(),
            'collided': self.collided,
        })


        if self._big_cam:
            result.update({
                'big_cam': carla_img_to_np(self.big_cam_image),
            })

        return result

    def apply_control(self, control=None, move_peds=True):
        """
        Applies very naive pedestrian movement.
        """
        if control is not None:
            self._player.apply_control(control)

        return {
            't': self._tick,
            'wall': time.time() - self._time_start,
            'ran_light': self.traffic_tracker.ran_light
        }

    def clean_up(self):
        for vehicle in self._actor_dict['vehicle']:
            # continue
            vehicle.stop_dtcrowd()

        for controller in self._actor_dict['ped_controller']:
            controller.stop()

        for sensor in self._actor_dict['sensor']:
            sensor.destroy()

        for actor_type in list(self._actor_dict.keys()):
            self._client.apply_batch([carla.command.DestroyActor(x) for x in
                                      self._actor_dict[actor_type]])
            self._actor_dict[actor_type].clear()

        self._actor_dict.clear()

        self._tick = 0
        self._time_start = time.time()

        if self._player:
            self._player.stop_dtcrowd()
        self._player = None

        # Clean-up cameras
        if self._rgb_queue:
            with self._rgb_queue.mutex:
                self._rgb_queue.queue.clear()

        if self._seg_queue:
            with self._seg_queue.mutex:
                self._seg_queue.queue.clear()

        if self._big_cam_queue:
            with self._big_cam_queue.mutex:
                self._big_cam_queue.queue.clear()

    @property
    def pedestrians(self):
        return self._actor_dict.get('pedestrian', [])

    @property
    def ped_controllers(self):
        return self._actor_dict.get('ped_controller', [])

    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        # Camera.
        self._rgb_queue = queue.Queue()
        self._seg_queue = queue.Queue()

        if self._big_cam:
            self._big_cam_queue = queue.Queue()
            rgb_camera_bp = self._blueprints.find('sensor.camera.rgb')
            rgb_camera_bp.set_attribute('image_size_x', '800')
            rgb_camera_bp.set_attribute('image_size_y', '600')
            rgb_camera_bp.set_attribute('fov', '120')
            big_camera = self._world.spawn_actor(
                rgb_camera_bp,
                carla.Transform(carla.Location(x=1.0, z=1.4),
                                carla.Rotation(pitch=0)),
                attach_to=self._player)
            big_camera.listen(self._big_cam_queue.put)
            self._actor_dict['sensor'].append(big_camera)

        rgb_camera_bp = self._blueprints.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', '384')
        rgb_camera_bp.set_attribute('image_size_y', '160')
        rgb_camera_bp.set_attribute('fov', '120')
        rgb_camera = self._world.spawn_actor(
            rgb_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4),
                            carla.Rotation(pitch=0)),
            attach_to=self._player)

        rgb_camera.listen(self._rgb_queue.put)
        self._actor_dict['sensor'].append(rgb_camera)

        seg_camera_bp = self._blueprints.find(
            'sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', '384')
        seg_camera_bp.set_attribute('image_size_y', '160')
        seg_camera_bp.set_attribute('fov', '120')
        self.seg_camera = self._world.spawn_actor(
            seg_camera_bp,
            carla.Transform(carla.Location(x=2.0, z=1.4),
                            carla.Rotation(pitch=0)),
            attach_to=self._player)

        self.seg_camera.listen(self._seg_queue.put)
        self._actor_dict['sensor'].append(self.seg_camera)

        calibration = np.identity(3)
        calibration[0, 2] = 384 / 2.0
        calibration[1, 2] = 160 / 2.0
        calibration[0, 0] = calibration[1, 1] = 384 / (2.0 * np.tan(120 * np.pi / 360.0))
        self.seg_camera.calibration = calibration

        #
        # image_w = seg_camera_bp.get_attribute("image_size_x").as_int()
        # image_h = seg_camera_bp.get_attribute("image_size_y").as_int()
        # fov = seg_camera_bp.get_attribute("fov").as_float()
        # focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # # In this case Fx and Fy are the same since the pixel aspect
        # # ratio is 1
        # K = np.identity(3)
        # K[0, 0] = K[1, 1] = focal
        # K[0, 2] = image_w / 2.0
        # K[1, 2] = image_h / 2.0
        #
        # def inv_rotate_vector(p, rot):
        #     yaw = rot.yaw
        #     cy = np.cos(np.rad2deg(yaw))
        #     sy = np.sin(np.rad2deg(yaw))
        #
        #     roll = rot.roll
        #     cr = np.cos(np.rad2deg(roll))
        #     sr = np.sin(np.rad2deg(roll))
        #
        #     pitch = rot.pitch
        #     cp = np.cos(np.rad2deg(pitch))
        #     sp = np.sin(np.rad2deg(pitch))
        #
        #     out_point = np.array([
        #         p[0] * (cp * cy) + p[1] * (cp * sy) + p[2] * sp,
        #         p[0] * (cy * sp * sr - sy * cr) + p[1] * (sy * sp * sr + cy * cr) + p[2] * (-cp * sr),
        #         p[0] * (-cy * sp * cr - sy * sr) + p[1] * (-sy * sp * cr + cy * sr) + p[2] * (cp * cr)
        #     ])
        #
        #     return out_point
        #
        # def inv_transform_point(point, rot, loc):
        #     out_point = point.copy()
        #     out_point -= loc
        #     return inv_rotate_vector(out_point, rot)
        #
        # def get_inv_matrix(rot, loc):
        #     yaw = rot.yaw
        #     cy = np.cos(np.rad2deg(yaw))
        #     sy = np.sin(np.rad2deg(yaw))
        #
        #     roll = rot.roll
        #     cr = np.cos(np.rad2deg(roll))
        #     sr = np.sin(np.rad2deg(roll))
        #
        #     pitch = rot.pitch
        #     cp = np.cos(np.rad2deg(pitch))
        #     sp = np.sin(np.rad2deg(pitch))
        #
        #     a = carla.Vector3D([0, 0, 0])
        #     inv_transform_point(a, rot, loc)
        #
        #     out = np.array([
        #                    [cp * cy, cp * sy, sp, a[0]],
        #                    [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr, a[1]],
        #                    [-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr, a[2]],
        #                    [0, 0, 0, 1]])
        #
        #     return out

        # tr = seg_camera.get_transform()
        # print("INVERSE CAM", get_inv_matrix(tr.rotation, tr.location))

        # Collisions.
        self.collided = False
        self._collided_frame_number = -1

        collision_sensor = self._world.spawn_actor(
            self._blueprints.find('sensor.other.collision'),
            carla.Transform(), attach_to=self._player)
        collision_sensor.listen(
            lambda event: self.__class__._on_collision(weakref.ref(self),
                                                       event))
        self._actor_dict['sensor'].append(collision_sensor)

        # Lane invasion.
        self.invaded = False
        self._invaded_frame_number = -1

        invasion_sensor = self._world.spawn_actor(
            self._blueprints.find('sensor.other.lane_invasion'),
            carla.Transform(), attach_to=self._player)
        invasion_sensor.listen(
            lambda event: self.__class__._on_invasion(weakref.ref(self),
                                                      event))
        self._actor_dict['sensor'].append(invasion_sensor)

    @staticmethod
    def _on_collision(weakself, event):
        _self = weakself()

        if not _self:
            return

        impulse = event.normal_impulse
        intensity = np.linalg.norm([impulse.x, impulse.y, impulse.z])

        if intensity > _self.col_threshold:
            _self.collided = True
            _self._collided_frame_number = event.frame_number

    @staticmethod
    def _on_invasion(weakself, event):
        _self = weakself()

        if not _self:
            return

        _self.invaded = True
        _self._invaded_frame_number = event.frame_number

    def __enter__(self):
        set_sync_mode(self._client, True)

        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self.clean_up()

        set_sync_mode(self._client, False)

    def render_world(self):
        return map_utils.render_world()

    def world_to_pixel(self, pos):
        return map_utils.world_to_pixel(pos)
