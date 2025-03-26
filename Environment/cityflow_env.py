from cityflow import Engine
import numpy as np
import json
import sys
import os
import gym
from Environment.utils.log import log
from Environment.utils.utils import gzip_file


import numpy as np
import sys
import time


class TrafficSignal:
    """
    Traffic Signal Control
    """

    def __init__(self, ID, eng, yellow_time, roadnet_info):
        self.roadnet_info = roadnet_info
        self.ID = ID
        self.eng = eng
        self.now_idx = 1
        self.old_idx = 1
        self.next_idx = None
        self.now_time = -1
        self.yellow_flag = False
        self.flicker = False
        self.yellow_time = yellow_time
        self.phase_number = len(roadnet_info["phases"])
        self.green_times = np.zeros(len(roadnet_info["roadlinks"]))
        self.phase_skips = np.zeros(len(roadnet_info["phases"]))
        self.green_skips = np.zeros(len(roadnet_info["roadlinks"]))
        self._set_observation_space()
        self.action_space = self.phase_number - 1
        self.set_eng_phase()

    def _set_observation_space(self):
        """
        include intersection dimension.
        TSflow, TSwait, TSgreen, TStime means array shape
        TSphase means how many phases available and roadlinks opened in
            each phase, and when in yellow phase, returns next controllable
            phase.
        TStime: phase activated time of every lane
        """
        self.observation_space = {
            "TSflow": np.zeros(len(self.roadnet_info["roadlinks"])),
            "TSwait": np.zeros(len(self.roadnet_info["roadlinks"])),
            "TSgreen": np.zeros(len(self.roadnet_info["roadlinks"])),
            "TSphase": np.array(self.roadnet_info["phases"][1:]),
            "TStime": np.array([1]),
            "Envtime": np.array([1]),
            "LaneCount": np.zeros(len(self.roadnet_info["roadlinks"])),
            "RoadLinkDirection": np.array(
                [x["type"] for x in self.roadnet_info["roadlinks"]]
            ),
        }

    def set_eng_phase(self):
        self.eng.set_tl_phase(self.ID, self.now_idx)

    def set_signal(self, action, action_pattern):
        if self.yellow_flag:
            # in yellow phase
            if self.now_time >= self.yellow_time:  # yellow time reached
                self.now_idx = self.next_idx
                self.set_eng_phase()
                self.yellow_flag = False
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_idx = self.now_idx
                elif action == 1:  # change to the next phase
                    self.next_idx = self.now_idx + 1
                    if self.next_idx == self.phase_number:
                        self.next_idx = 1
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                self.next_idx = action + 1

            # set phase
            if self.now_idx == self.next_idx:  # light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.now_idx = 0
                self.set_eng_phase()
                self.yellow_flag = True

                for i in range(self.phase_number):
                    if i != self.old_idx:
                        self.phase_skips[i] += 1
                self.phase_skips[self.next_idx] = 0
                green_ons = set(
                    self.roadnet_info["phases"][self.old_idx]
                    + self.roadnet_info["phases"][self.next_idx]
                )
                self.green_skips += 1
                for g in green_ons:
                    self.green_skips[g] = 0

    def step_time(self):
        if self.now_idx != self.old_idx:
            self.now_time = 1
        else:
            self.now_time += 1

    def _get_signal_state(self):
        if self.now_idx == 0:
            return self.next_idx - 1
        return self.now_idx - 1

    def get_state(self):
        TSflow = np.zeros(len(self.roadnet_info["roadlinks"]))
        TSwait = np.zeros(len(self.roadnet_info["roadlinks"]))
        TSgreen = np.zeros(len(self.roadnet_info["roadlinks"]))
        TSphase = self._get_signal_state()
        TStime = self.now_time
        LaneCount = np.zeros(len(self.roadnet_info["roadlinks"]))
        for roadlink in range(len(self.roadnet_info["roadlinks"])):
            count = self.eng.get_roadlink_vehicle_count(self.ID, roadlink)
            wait_count = self.eng.get_roadlink_waiting_vehicle_count(self.ID, roadlink)
            assert count[1] == wait_count[1], f"{count} {wait_count}"
            if count[1] == 0:
                TSflow[roadlink] = 0
                TSwait[roadlink] = 0
            else:
                TSflow[roadlink] = count[0] / count[1]
                TSwait[roadlink] = wait_count[0] / wait_count[1]
            LaneCount[roadlink] = count[1]
            TSgreen[roadlink] = 0
        for phase in self.roadnet_info["phases"][self.now_idx]:
            TSgreen[phase] = 1
        for i, l in enumerate(TSgreen):
            if l == 0:
                if self.green_times[i] > 0 and self.green_times[i] < 0:
                    pass
                    # constraint - has not spent enough time as green
                self.green_times[i] = 0
            else: self.green_times[i] += 1
        envtime = self.eng.get_current_time()
        return {
            "TSflow": TSflow,
            "TSwait": TSwait,
            "TSgreen": TSgreen,
            "TSphase": TSphase,
            "TStime": TStime,
            "Envtime": envtime,
            "LaneCount": LaneCount,
        }

    def update_old(self):
        self.old_idx = self.now_idx


class Intersection:
    def __init__(self, ID, config, eng, intersection_names, virtual_intersection_names):
        self.ID = ID
        self.config = config
        self.eng = eng
        self.intersection_names = intersection_names
        self.virtual_intersection_names = virtual_intersection_names
        self.save_lane_count_step = 5
        self.lanename2roaddirection = {}

        self._init_components()

        self._set_observation_space()
        self._set_action_space()

    def _init_components(self):
        self.roadnet_info = self.config["ROADNET_INFO"][self.ID]

        self.TS = TrafficSignal(
            self.ID, self.eng, self.config["YELLOW_TIME"], self.roadnet_info
        )
        self.last_reward = 0

        self.saved_phases = np.zeros((self.save_lane_count_step,), dtype=int)
        self.saved_phases[:] = -1

        self.lane_vehicle_count_mat = np.zeros(
            (  # this is in count
                self.save_lane_count_step,
                len(self.roadnet_info["connection"]),
                3,
            ),
            dtype=int,
        )
        self.lane_vehicle_out_count_mat = np.zeros(
            (self.save_lane_count_step, len(self.roadnet_info["connection"]), 3),
            dtype=int,
        )

    def reset(self):
        self._init_components()

    def _get_road_direction_idx(self, roadname, direction):
        for num, i in enumerate(self.roadnet_info["roadlinks"]):
            if i["startRoad"] == roadname and i["type"] == direction:
                return num
        raise ValueError(roadname, direction)

    """
       Intersection: CityFlow.list_intersections
       roadlinks: Intersection.[0..11] cross this intersection
       Roads: belong to intersection it start with. intersection.roadnames
       lanes: belongs to roads, order as DirectionNames

       DirectionNames: 3 direction name, [left, straight, right]
       RoadsOut: the terminal intersection of a road belongs to this 
           intersection that out of this intersection [4]
       RoadsIn: the start inter_x.RoadsOut_y of a road belongs to adjacent 
           intersection that goes into this intersection, [[x, y] * 12]
       RoadLinksOut: roadlink_i go out to which RoadsOut [12]
       RoadLinksIn: roadlink_i go in from which RoadsIn [12]
       RoadOutLanes: lane number of roadout_i [4 * 3]
       LaneCount: roadlink_i lane number
       InLaneVehicleNumber: vehicle number that has entered the lane, [4, 3],
           [i, j] means RoadsIn[i] and DirectionNames[j]
       OutLaneVehicleNumber: vehicle number that has leaved the lane, [4, 3]
           [i, j] means RoadsIn[i] and DirectionNames[j]
           *lane change will count. number is small, so ignore now.
       TSprevphases: previous phase actions, its shape
       RoadLinkDirection: the direction name of roadlink_i
    """

    def _set_observation_space(self):
        tot = self.TS.observation_space
        tot["DirectionNames"] = [
            "turn_left",
            "go_straight",
            "turn_right",
        ]  # now assume three direction
        self.lane_direction_number = len(tot["DirectionNames"])
        self.roadnames = list(self.roadnet_info["connection"].keys())
        self.roadnames.sort()
        tot["RoadsOut"] = []
        tot["RoadsIn"] = []
        for road in self.roadnames:
            interto = self.roadnet_info["connection"][road][0]
            intertoid = (
                self.intersection_names.index(interto)
                if interto in self.intersection_names
                else -1 - self.virtual_intersection_names.index(interto)
            )
            tot["RoadsOut"].append(intertoid)
        tot["RoadLinksOut"] = []
        tot["RoadLinksIn"] = []  # can't fill now, fill later
        tot["RoadOutLanes"] = [
            [0] * len(tot["DirectionNames"]) for _ in self.roadnames
        ]  # also can't fill now
        tot["LaneCount"] = []
        for rl in self.roadnet_info["roadlinks"]:
            tot["RoadLinksOut"].append(self.roadnames.index(rl["endRoad"]))
            tot["LaneCount"].append(rl["lanenumber"])
        tot["InLaneVehicleNumber"] = [
            self.save_lane_count_step,
            len(self.roadnames),
            self.lane_direction_number,
        ]
        tot["OutLaneVehicleNumber"] = [
            self.save_lane_count_step,
            len(self.roadnames),
            self.lane_direction_number,
        ]
        tot["TSprevphases"] = [self.save_lane_count_step]
        self.observation_space = tot

    def _set_road_links_in(self, roadname2id):
        self.roadinnames = []
        rlin = self.observation_space["RoadLinksIn"]
        rin = self.observation_space["RoadsIn"]
        for rl in self.roadnet_info["roadlinks"]:
            rn = rl["startRoad"]
            if rn not in roadname2id:
                rlin.append(-1)
            else:
                if rn not in self.roadinnames:
                    self.roadinnames.append(rn)
                    rin.append(roadname2id[rn])
                rlin.append(self.roadinnames.index(rn))

    def _get_road_id(self, inters, ask_inter_id, ask_roadnumber):
        torid = -1
        for num, [i, j] in enumerate(self.observation_space["RoadsIn"]):
            if i >= 0 and ask_inter_id == inters[i].ID and ask_roadnumber == j:
                torid = num
        if torid != -1:
            return torid
        torid = []
        for num, [i, j] in enumerate(self.observation_space["RoadsIn"]):
            if i < 0 and ask_roadnumber == j:
                torid.append([i, num])
        return torid

    def _update_lanename(
        self,
        lanename2roaddirection,
        inters,
        now_in_lane_vehicles,
        last_in_lane_vehicles,
    ):
        self.lanename2roaddirection = lanename2roaddirection
        self.now_in_lane_vehicles = now_in_lane_vehicles
        self.last_in_lane_vehicles = last_in_lane_vehicles
        for rn, [rname, toid] in enumerate(
            zip(self.roadnames, self.observation_space["RoadsOut"])
        ):
            if toid < 0:
                continue
            o_inter = inters[toid]
            torid = -1
            for num, [i, j] in enumerate(o_inter.observation_space["RoadsIn"]):
                if i >= 0 and self == inters[i] and rn == j:
                    torid = num
            assert torid == o_inter._get_road_id(inters, self.ID, rn)
            now_lanen = 0
            for dn, d in enumerate(self.observation_space["RoadOutLanes"][rn]):
                # now assume lane order same as direction name order
                for _ in range(d):
                    lanename = "%s_%d" % (rname, now_lanen)
                    now_lanen += 1
                    self.lanename2roaddirection[lanename] = [toid, torid, dn]
                    self.now_in_lane_vehicles[lanename] = set()
                    self.last_in_lane_vehicles[lanename] = set()

    def _set_action_space(self):
        tot = [self.TS.action_space]
        self.action_space = tot

    def set_signal(self, action, action_pattern):
        self.TS.set_signal(action, action_pattern)

    def step_time(self):
        self.TS.step_time()

    def get_state(self):
        res = self.TS.get_state()
        res["InLaneVehicleNumber"] = self.lane_vehicle_count_mat
        res["OutLaneVehicleNumber"] = self.lane_vehicle_out_count_mat
        self.saved_phases[-1] = res["TSphase"]
        res["TSprevphases"] = list(self.saved_phases)
        res["pressure"] = 0  # pressure is too expensive to calculate rn
        return res

    def act(self, action, pattern):
        arr = []
        for i in self.action_space:
            arr.append(action % i)
            action //= i
        actions = arr
        self.set_signal(actions[0], pattern)
        assert len(actions) == 1

    def get_default_action(self):
        # use single number format as default action
        return [0]

    def _get_pressure_observation(self):
        lane_count_1 = self.eng.get_lane_vehicle_count()
        lane_count_2 = self.eng.get_lane_waiting_vehicle_count()
        res = []
        for lanelink in self.roadnet_info["roadlinks"]:
            # Aggregate RA and RS for the current roadlink without repeated lookups
            RA = sum(lane_count_1[s] for s, _ in lanelink["lanelinks"])
            RS = sum(lane_count_2[e] for _, e in lanelink["lanelinks"])

            # Calculate the difference and append to results
            RR = RA - RS
            res.append(RR)
        return np.array(res)

    def _get_average_vehicle_count(self, type, weight):
        res = 0
        cres = 0
        if type.lower() == "flow":
            target_func = self.eng.get_roadlink_vehicle_count
        elif type.lower() == "wait":
            target_func = self.eng.get_roadlink_waiting_vehicle_count
        else:
            raise ValueError("unknown type " + type)
        for roadlink in range(len(self.roadnet_info["roadlinks"])):
            count = target_func(self.ID, roadlink)
            if count[1] == 0:
                pass
            else:
                res += count[0] / count[1]
                cres += 1
        return -res / cres * weight

    def _get_average_pressure(self, type, weight):
        lane_count = self.eng.get_lane_vehicle_count()
        if type.lower() == "intersection":
            # count intersection as one unit, |Sigma(#in) - Sigma(#out)|
            lanelinks = []
            for roadlink in self.roadnet_info["roadlinks"]:
                lanelinks.extend(roadlink["lanelinks"])
            start, end = zip(*lanelinks)
            start = set(start)
            end = set(end)
            res = 0
            for s in start:
                res += lane_count[s]
            for e in end:
                res -= lane_count[e]
            return -abs(res) * weight
        elif type.lower() == "roadlink":
            # count every roadlink pressure and average, Sigma(|P_ri|)
            lanelinks = [x["lanelinks"] for x in self.roadnet_info["roadlinks"]]
            res = []
            for lanelink in lanelinks:
                start, end = zip(*lanelink)
                start = set(start)
                end = set(end)
                RR = 0
                for s in start:
                    RR += lane_count[s]
                for e in end:
                    RR -= lane_count[e]
                RR = abs(RR)
                res.append(RR)
            return -np.array(res).mean() * weight
        elif type.lower() == "lanelink":
            # count every lanelink pressure and average
            res = []
            for roadlink in self.roadnet_info["roadlinks"]:
                for s, t in roadlink["lanelinks"]:
                    res.append(abs(lane_count[s] - lane_count[t]))
            return -np.array(res).mean() * weight
        else:
            raise ValueError("unknown type " + type)

    def get_reward(self, weight):
        res = 0
        wkeys = list(weight.keys())
        for i in wkeys:
            if not weight[i]:
                del weight[i]
                continue
            weight[i] = float(weight[i])
            if not weight[i]:
                del weight[i]
        if "AVERAGE_FLOW_VEHICLE_COUNT" in weight:
            res += self._get_average_vehicle_count(
                type="flow", weight=weight["AVERAGE_FLOW_VEHICLE_COUNT"]
            )
        if "AVERAGE_WAIT_VEHICLE_COUNT" in weight:
            res += self._get_average_vehicle_count(
                type="wait", weight=weight["AVERAGE_WAIT_VEHICLE_COUNT"]
            )
        if "AVERAGE_INTERSECTION_PRESSURE" in weight:
            res += self._get_average_pressure(
                type="intersection", weight=weight["AVERAGE_INTERSECTION_PRESSURE"]
            )
        if "AVERAGE_ROADLINK_PRESSURE" in weight:
            res += self._get_average_pressure(
                type="roadlink", weight=weight["AVERAGE_ROADLINK_PRESSURE"]
            )
        if "AVERAGE_LANELINK_PRESSURE" in weight:
            res += self._get_average_pressure(
                type="lanelink", weight=weight["AVERAGE_LANELINK_PRESSURE"]
            )
        if self.config["DELTA_REWARD"]:
            res = res - self.last_reward
            self.last_reward += res
            res *= self.config["DELTA_REWARD_MULTIPLIER"]
        return res

    def update_old(self):
        self.TS.update_old()


class Flow:
    def __init__(self):
        pass


"""CityFlow Environment, split action into and combine data from Intersections

Args:
    log_path: a folder to save log.
    work_folder: where to read flowFile and roadnetFile.
    config: config file.
    logfile: if set, save cityflow engine log in this file.
    seed: engine seed.
    suffix: whether add suffix to files used or generated in engine. if all 
        files save in same folder, we should add suffix to avoid overwrite.
    predefined_flow: whether use predefined flow file. if set False, will use 
        flow generator to generate flow.
Functions:
    ???
"""


class CityFlowEnv:
    def __init__(self, log_path, work_folder, config, logfile="", seed=0, suffix=True):
        self.log_path = log_path
        self.work_folder = work_folder
        self.config = config
        self.logfile = logfile
        self.seed = seed
        self.spec = None
        self.config_size = 0
        self.env_padding = "ENV_PADDING" in config and config["ENV_PADDING"]
        date_time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        folder_name = "%s_%d" % (date_time_str, seed)
        self.log_path = os.path.join(self.log_path, folder_name)
        os.makedirs(self.log_path, exist_ok=True)

        file_suffix = ""
        if suffix:
            file_suffix = "_" + str(seed)

        self.replay_path = (
            os.path.join(self.log_path, "replay%s.txt" % file_suffix) + ".%04d"
        )
        self.replay_count = 0
        config_dict = {
            "interval": self.config["INTERVAL"],
            "seed": seed,
            "dir": "",
            "roadnetFile": os.path.join(self.work_folder, self.config["ROADNET_FILE"]),
            "flowFile": os.path.join(self.work_folder, self.config["FLOW_FILE"]),
            "rlTrafficLight": True,
            "laneChange": True,
            "saveReplay": self.config["SAVEREPLAY"],
            "roadnetLogFile": os.path.join(
                self.log_path, "roadnet%s.json" % file_suffix
            ),
            "replayLogFile": self.replay_path % self.replay_count,
        }

        if "linux" not in sys.platform:
            print(
                "[WARN ] not in linux platform, some log from CityFlowEnv "
                "can't be recorded in log file!"
            )
        config_path = os.path.join(self.log_path, "cityflow_config.json")
        self.config_path = config_path
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(config_dict, f)
                self.log("dump cityflow config:", config_path, level="TRACE")
            self.config_size = os.path.getsize(config_path)

        if len(logfile) > 0:
            logfile = os.path.join(log_path, "%s%s" % (logfile, file_suffix))
            self.logfile = logfile
        self.init_engine()

        self.list_inter_log = None

        # check min action time
        if self.config["MIN_ACTION_TIME"] <= self.config["YELLOW_TIME"]:
            self.log("MIN_ACTION_TIME should include YELLOW_TIME", level="ERROR")
            pass
            # raise ValueError

        roadnet_info_keys = list(self.config["ROADNET_INFO"].keys())
        roadnet_info_keys.sort()
        virtual_inters = self.config["VIRTUAL_INTERSECTION_NAMES"]
        virtual_inter_keys = list(virtual_inters.keys())
        virtual_inter_keys.sort()
        self.virtual_road_out_lanes = np.zeros((len(virtual_inters), 3), int)
        self.virtual_direction_names = ["turn_left", "go_straight", "turn_right"]
        self.list_intersection = [
            Intersection(
                x, self.config, self.eng, roadnet_info_keys, virtual_inter_keys
            )
            for x in roadnet_info_keys
        ]
        for inter in self.list_intersection:
            assert (
                self.virtual_direction_names
                == inter.observation_space["DirectionNames"]
            )
        self.intername2idx = {}
        self.roadname2id = {}
        self.lanename2roaddirection = {}
        self.now_in_lane_vehicles = {}
        self.last_in_lane_vehicles = {}
        self.all_vehicles = set()
        self.wait_ticks = 0
        for num, inter in enumerate(self.list_intersection):
            self.intername2idx[inter.ID] = num
            assert roadnet_info_keys[num] == inter.ID
            for rnum, rname in enumerate(inter.roadnames):
                self.roadname2id[rname] = [num, rnum]
        for ID in virtual_inters:
            v_inter = virtual_inters[ID]
            num = virtual_inter_keys.index(ID)
            roadnames = list(v_inter["connection"].keys())
            assert len(roadnames) == 1
            self.roadname2id[roadnames[0]] = [-1 - num, 0]
        for inter in self.list_intersection:
            inter._set_road_links_in(self.roadname2id)
            for rl in inter.roadnet_info["roadlinks"]:
                if rl["startRoad"] in self.roadname2id:
                    num, rnum = self.roadname2id[rl["startRoad"]]
                    if num < 0:
                        vid = -1 - num
                        typeid = self.virtual_direction_names.index(rl["type"])
                        self.virtual_road_out_lanes[vid, typeid] += rl["lanenumber"]
                        continue
                    ninter = self.list_intersection[num]
                    ninter.observation_space["RoadOutLanes"][rnum][
                        ninter.observation_space["DirectionNames"].index(rl["type"])
                    ] += rl["lanenumber"]
        for inter in self.list_intersection:
            inter._update_lanename(
                self.lanename2roaddirection,
                self.list_intersection,
                self.now_in_lane_vehicles,
                self.last_in_lane_vehicles,
            )
            # self.log(inter.ID, inter.observation_space)
        self.update_virtual_lanename(virtual_inters, virtual_inter_keys)
        # self.log(self.lanename2roaddirection)

        self.list_inter_log = [[] for i in range(len(self.list_intersection))]

        self._set_adj_mat()
        self._set_observation_space()
        self._set_action_space()

    def update_virtual_lanename(self, virtual_inters, virtual_inter_keys):
        for v_num, v_inter_key in enumerate(virtual_inter_keys):
            v_inter = virtual_inters[v_inter_key]
            rn = 0
            rname = list(v_inter["connection"].keys())[0]
            toiname = v_inter["connection"][rname][0]
            toid = -999
            for num, inter in enumerate(self.list_intersection):
                if inter.ID == toiname:
                    toid = num
                    break
            assert toid >= 0
            o_inter = self.list_intersection[toid]
            torid = o_inter._get_road_id(self.list_intersection, self, rn)
            assert isinstance(torid, list)
            for i, j in torid:
                if virtual_inter_keys[-1 - i] == v_inter_key:
                    torid = j
            now_lanen = 0
            dn = 0
            for dn, d in enumerate(self.virtual_road_out_lanes[v_num]):
                for _ in range(d):
                    lanename = "%s_%d" % (rname, now_lanen)
                    now_lanen += 1
                    self.lanename2roaddirection[lanename] = [toid, torid, dn]
                    self.now_in_lane_vehicles[lanename] = set()
                    self.last_in_lane_vehicles[lanename] = set()

    def log(self, *argv, **kwargs):
        level = "INFO" if "level" not in kwargs else kwargs["level"]
        if level in ["INFO", "WARN", "ERROR"]:
            print("[%-5s]" % level, *argv)

    def init_engine(self):
        time.sleep(1)
        self.eng = Engine(self.config_path, 4)

    def _set_adj_mat(self):
        n = len(self.list_intersection)
        self.adj_mat = np.zeros((n, n))
        self.adj_mat[:] = 1e100
        for i in self.config["ROADNET_INFO"].keys():
            if i not in self.intername2idx.keys():
                continue
            iidx = self.intername2idx[i]
            conn = self.config["ROADNET_INFO"][i]["connection"]
            for road in conn:
                j, dist = conn[road]
                if j not in self.intername2idx.keys():
                    continue
                jidx = self.intername2idx[j]
                self.adj_mat[iidx, jidx] = dist
        for i in range(n):
            self.adj_mat[i, i] = 0

    def _set_observation_space_one(self, i):
        observation_space = self.list_intersection[i].observation_space
        return observation_space

    def _set_observation_space(self):
        self.observation_space = []
        for i in range(len(self.list_intersection)):
            self.observation_space.append(self._set_observation_space_one(i))
        self.observation_space_keys = list(self.observation_space[0].keys())
        """
        for obs in self.observation_space:
            if obs != self.observation_space[0] and self.env_padding and False:
                self.log('observation space not all same, and env_padding is '
                         'set, which is not implemented!', level = 'ERROR')
                raise ValueError
        """
        self.observation_space = {
            "intersections": self.observation_space,
            "virtual_intersection_out_lines": self.virtual_road_out_lanes,
            "adj_mat": self.adj_mat,
        }

    def _set_action_space_one(self, i):
        return self.list_intersection[i].action_space

    def _set_action_space(self):
        self.action_space = [
            gym.spaces.Discrete(self._set_action_space_one(i)[0])
            for i in range(len(self.list_intersection))
        ]

        """
        for act in self.action_space:
            if act != self.action_space[0] and self.env_padding:
                self.log('action space not all same, and env_padding is set, '
                         'which is not implemented!', level = 'ERROR')
                raise ValueError
        """

    def reset(self):
        self.eng.reset()
        self.replay_count += 1

        # reset intersections (grid)
        for inter in self.list_intersection:
            inter.reset()

        # get new measurements
        for inter in self.list_intersection:
            inter.step_time()

        # self.flow_generator.reset()

        for k in self.now_in_lane_vehicles:
            self.now_in_lane_vehicles[k] = set()
        for k in self.last_in_lane_vehicles:
            self.last_in_lane_vehicles[k] = set()
        self.all_vehicles.clear()
        self.wait_ticks = 0

        state = self._collect_state()
        return state, {"average_time": 0.0, "average_delay": 0.0}

    def _collect_state(self):
        res = []
        for inter in self.list_intersection:
            res.append(inter.get_state())
            # self.log(inter.ID, res[-1]['InLaneVehicleNumber'], level='TRACE')
        return res

    def _collect_reward(self):
        rew = []
        for inter in self.list_intersection:
            rew.append(inter.get_reward(self.config["REWARD_INFO"]))
        return np.array(rew)

    def _is_done(self):
        return self.eng.get_current_time() >= self.config["EPISODE_LEN"]

    def _average_time(self):
        return self.eng.get_average_travel_time()

    def _average_delay(self):
        return self.eng.get_average_delay()

    def _average_road_delay(self):
        return self.eng.get_road_average_delay()

    def get_green_times(self):
        return [
                np.mean([(gt < 40) for l, gt in enumerate(inter.TS.green_times) if gt > 0 and l not in {2, 3, 6, 10}])
                for inter in self.list_intersection
            ]
      
    def get_phase_skips(self):
        return [
                np.mean([(ps < 16) for l, ps in enumerate(inter.TS.phase_skips) if l != 0])
                for inter in self.list_intersection
            ]

    def get_green_skips(self):
        return [
                np.mean([(gs < 4) for l, gs in enumerate(inter.TS.green_skips) if l not in {2, 3, 6, 10}])
                for inter in self.list_intersection
            ]

    def step(self, action):
        if self.config["ACTION_PATTERN"] == "switch":
            raise NotImplementedError("ACTION_PATTERN `switch` " "is not implemented")

        all_reward = np.zeros(len(self.list_intersection), dtype="float")
        for i in range(self.config["MIN_ACTION_TIME"]):
            if i == 0:
                self.step_lane_vehicles()
            self._inner_step(action, self.config["ACTION_PATTERN"])
            state = self._collect_state()
            all_reward += self._collect_reward()
            done = self._is_done()

        all_reward /= self.config["MIN_ACTION_TIME"]

        infos = {
            "average_time": self._average_time(),
            "average_delay": self._average_delay(),
            "current_time": self.eng.get_current_time(),
            "throughput": (len(self.all_vehicles) - len(self.eng.get_vehicles(True))),
            "average_wait_time": (
                self.wait_ticks / len(self.all_vehicles)
                if len(self.all_vehicles)
                else -1
            ),
            "individual_rewards": all_reward,
            "total_reward": all_reward.sum(),
            "green_times": np.mean(self.get_green_times()),
            "phase_skips": np.mean(self.get_phase_skips()),
            "green_skips": np.mean(self.get_green_skips()),
            "individual_green_times": np.array(self.get_green_times()),
            "individual_phase_skips": np.array(self.get_phase_skips()),
            "individual_green_skips": np.array(self.get_green_skips()),
        }
        return state, all_reward.mean(), done, infos

    def step_lane_vehicles(self):
        for inter in self.list_intersection:
            inter.lane_vehicle_count_mat[:-1] = inter.lane_vehicle_count_mat[1:]
            inter.lane_vehicle_count_mat[-1] = 0
            inter.lane_vehicle_out_count_mat[:-1] = inter.lane_vehicle_out_count_mat[1:]
            inter.lane_vehicle_out_count_mat[-1] = 0
            inter.saved_phases[:-1] = inter.saved_phases[1:]

    def update_lane_vehicles(self, lane_vehicles):
        for i in self.now_in_lane_vehicles:  # copy current to last
            self.last_in_lane_vehicles[i] = self.now_in_lane_vehicles[i].copy()
            self.now_in_lane_vehicles[i].clear()

        # for i in self.now_in_lane_vehicles:
        # self.now_in_lane_vehicles[i].clear()
        for k, vs in lane_vehicles.items():
            if k in self.lanename2roaddirection:
                for v in vs:
                    if "shadow" in v:
                        continue
                    self.now_in_lane_vehicles[k].add(v)

        for k in self.now_in_lane_vehicles:
            i, r, d = self.lanename2roaddirection[k]
            inter = self.list_intersection[i]
            inter.lane_vehicle_count_mat[-1, r, d] += len(
                self.now_in_lane_vehicles[k] - self.last_in_lane_vehicles[k]
            )
            inter.lane_vehicle_out_count_mat[-1, r, d] += len(
                self.last_in_lane_vehicles[k] - self.now_in_lane_vehicles[k]
            )

    def _inner_step(self, actions, pattern):
        for action, inter in zip(actions, self.list_intersection):
            inter.update_old()
            inter.act(action, pattern)
        for i in range(int(1 / self.config["INTERVAL"])):
            # self.flow_generator.check(self.eng.get_current_time())
            self.eng.next_step()  # catch errors and report to above
            if (
                "NOT_COUNT_VEHICLE_VOLUME" not in self.config
                or not self.config["NOT_COUNT_VEHICLE_VOLUME"]
            ):
                lane_vehicles = self.eng.get_lane_vehicles()
                vehicles = self.eng.get_vehicles(include_waiting=True)
                for v in vehicles:
                    self.all_vehicles.add(v)
                self.wait_ticks += len(vehicles)
                v_speeds = self.eng.get_vehicle_speed()
                for v in v_speeds:
                    if v_speeds[v] > 0.1:
                        self.wait_ticks -= 1
                self.update_lane_vehicles(lane_vehicles)
        for inter in self.list_intersection:
            inter.step_time()

    def get_default_action(self):
        return [x.get_default_action() for x in self.list_intersection]
