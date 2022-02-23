from itertools import zip_longest
import json
from collections import defaultdict

import numpy as np
from habitat import Env, logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm, trange
import json 
import ipdb
from vlnce_baselines.common.environments import VLNCEInferenceEnv
from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
import habitat_sim 

def evaluate_agent(config: Config) -> None:
    split = config.EVAL.SPLIT
    data_path = config.EVAL.DATA_PATH
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    config.TASK_CONFIG.TASK.SENSORS = []
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.DATA_PATH = data_path
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.freeze()

    num_nan = 0
    path_data = {}
    
    env = Env(config=config.TASK_CONFIG)

    # assert config.EVAL.NONLEARNING.AGENT in [
    #     "RandomAgent",
    #     "HandcraftedAgent",
    # ], "EVAL.NONLEARNING.AGENT must be either RandomAgent or HandcraftedAgent."

    if config.EVAL.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomAgent()
    elif config.INFERENCE.NONLEARNING.AGENT == "HandcraftedAgent":
        agent = HandcraftedAgent()
    else:
        agent = GridToSimAgent(config.EVAL.NONLEARNING, env)
    stats = defaultdict(float)
    no_reachable_eps = []
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))
    for _ in trange(num_episodes):
        obs = env.reset()
        agent.reset()
        path_data[env.current_episode.episode_id] = []
      
        while not env.episode_over:

            action ,is_nan, ep_id = agent.act(obs)
            if is_nan:
                no_reachable_eps.append(ep_id)
            obs = env.step(action)
            path_data[env.current_episode.episode_id].append(env._sim.get_agent_state().position.tolist())
        if is_nan:
            num_nan += 1
        for m, v in env.get_metrics().items():
            stats[m] += v
    print("NAN", num_nan)
    print("NO reachable eps ", set(no_reachable_eps))
    print("Num ", len(set(no_reachable_eps)))
    stats = {k: v / num_episodes for k, v in stats.items()}

    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(f"stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json", "w") as f:
        json.dump(stats, f, indent=4)

    with open("glove_val_seen_result_sim_loc.json","w") as f:
        json.dump(path_data,f)
    print("DONE !!")


def nonlearning_inference(config: Config) -> None:
    split = config.INFERENCE.SPLIT
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    config.TASK_CONFIG.DATASET.SPLIT = config.INFERENCE.SPLIT
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.TASK.MEASUREMENTS = []
    config.TASK_CONFIG.TASK.SENSORS = []
    config.freeze()

    env = VLNCEInferenceEnv(config=config)
    
    assert config.INFERENCE.NONLEARNING.AGENT in [
        "RandomAgent",
        "HandcraftedAgent",
    ], "INFERENCE.NONLEARNING.AGENT must be either RandomAgent or HandcraftedAgent."
    
    if config.INFERENCE.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomAgent()
    elif config.INFERENCE.NONLEARNING.AGENT == "HandcraftedAgent":
        agent = HandcraftedAgent()
    else:
        agent = GridToSimAgent(config.INFERENCE.NONLEARNING.AGENT.CONFIG)


    episode_predictions = defaultdict(list)
    for _ in tqdm(range(len(env.episodes)), desc=f"[inference:{split}]"):
        env.reset()
        obs = agent.reset()

        episode_id = env.current_episode.episode_id
        episode_predictions[episode_id].append(env.get_info(obs))

        while not env.get_done(obs):
            obs = env.step(agent.act(obs))
            episode_predictions[episode_id].append(env.get_info(obs))

    with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
        json.dump(episode_predictions, f, indent=2)

    logger.info(f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}")


class RandomAgent(Agent):
    """Selects an action at each time step by sampling from the oracle action
    distribution of the training set.
    """

    def __init__(self, probs=None):
        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = [0.02, 0.68, 0.15, 0.15]

    def reset(self):
        pass

    def act(self, observations):
        return {"action": np.random.choice(self.actions, p=self.probs)}


class HandcraftedAgent(Agent):
    """Agent picks a random heading and takes 37 forward actions (average
    oracle path length) before calling stop.
    """

    def __init__(self,):
        self.reset()

    def reset(self):
        # 9.27m avg oracle path length in Train.
        # Fwd step size: 0.25m. 9.25m/0.25m = 37
        self.forward_steps = 37
    
        self.turns = np.random.randint(0, int(360 / 15) + 1)

    def act(self, observations):
        if self.turns > 0:
            self.turns -= 1
            return {"action": HabitatSimActions.TURN_RIGHT}
        if self.forward_steps > 0:
            self.forward_steps -= 1
            return {"action": HabitatSimActions.MOVE_FORWARD}
        return {"action": HabitatSimActions.STOP}
    

class GridToSimAgent(Agent):

    def __init__(self, config, env):
      
        with open(config.RESULT_PATH, "r") as f:
            self.data = json.load(f)
        
        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        
        self.shortest_path_follower = ShortestPathFollowerCompat(sim = env._sim,
                                                                 goal_radius = 0.25, 
                                                                return_one_hot = False)
        self.path_finder = env._sim.pathfinder
        self.env = env
        self.tmp_goal_index = 1
        self.is_nan  = False
        
        

    def reset(self):
        self.tmp_goal_index = 1 
        self.is_nan = False

    def act(self, observations):

        episode_id = self.env.current_episode.episode_id
        ep_results  = self.data[episode_id]
        sim_path = ep_results["sim_path"]
        start_position = self.env._sim.get_agent_state().position
        mid_value = start_position[1]
        index = -1
        tmp_goal = sim_path[index]
        tmp_goal[1] = mid_value

        back_step = 0
        is_search = False 
        while self.path_finder.is_navigable(np.array(tmp_goal)):
            index -= 1
            tmp_goal = sim_path[index]
            tmp_goal[1] = mid_value
            back_step += 1
            
            if abs(index) >= len(sim_path) or \
                (back_step > 10 and not self.path_finder.is_navigable(np.array(tmp_goal))):
                is_search = True 
                break 
        print(index)
        if is_search:
            if not self.path_finder.is_navigable(np.array(tmp_goal)):
                #tmp_goal = self.path_finder.snap_point(np.array(tmp_goal)) 
                #if not tmp_goal == tmp_goal:

                t_g = sim_path[-1].copy()

                x = np.linspace(-3.0, 3.0, 61)
                y = np.linspace(-3.0, 3.0, 61)

                z = np.linspace(-1,1, 11)
                
                # try like as a ball 
                #xv, zv ,yv= np.meshgrid(x,z,y)
                #a = list(zip(xv.flatten(), zv.flatten(), yv.flatten()))
                #candidatate_points = sorted(a, key=lambda k: k[0]*k[0] +k[1]*k[1]+k[2]*k[2] )
                xv, yv = np.meshgrid(x,y)
                a = list(zip(xv.flatten(), yv.flatten()))
                candidatate_points = sorted(a, key=lambda k: k[0]*k[0] +k[1]*k[1] )
                #while not self.path_finder.is_navigable(np.array(t_g)): 
                
                can_xy_points = []
                for x,y in candidatate_points:
                    t_g[0] += x
                    t_g[2] += y 
                    can_xy_points.append(t_g)
                    if self.path_finder.is_navigable(np.array(t_g)):
                        break 
                    else:
                        t_g = sim_path[-1].copy()

                if not self.path_finder.is_navigable(np.array(t_g)):
                    for points in can_xy_points:
                        x,y  = points[0],points[1]
                        for h in z:
                            p = [x,h,y]
                            if self.path_finder.is_navigable(np.array(p)):
                                t_g = p
                                break 
                        if self.path_finder.is_navigable(np.array(p)):
                            break
                    if not self.path_finder.is_navigable(np.array(t_g)):
                        self.is_nan = True

                tmp_goal = t_g
              
        act_index = self.shortest_path_follower.get_next_action(np.array(tmp_goal))
        if act_index is None:
            return {"action": self.actions[0]}, self.is_nan, episode_id
            
        # tmp_goal = sim_path[self.tmp_goal_index]
        # tmp_goal[1] = mid_value
        # if not self.path_finder.is_navigable(np.array(tmp_goal)):
        #     tmp_goal = self.path_finder.snap_point(np.array(tmp_goal)) 
        # act_index = self.shortest_path_follower.get_next_action(np.array(tmp_goal))
        # while act_index == None:
        #     self.tmp_goal_index += 1
        #     if self.tmp_goal_index > len(sim_path)-1:
        #         return {"action": self.actions[0]}
        #     tmp_goal = sim_path[self.tmp_goal_index]
        #     tmp_goal[1] = mid_value
        #     if not self.path_finder.is_navigable(np.array(tmp_goal)):
        #         tmp_goal = self.path_finder.snap_point(np.array(tmp_goal))
        #     act_index = self.shortest_path_follower.get_next_action(np.array(tmp_goal))
       
        return {"action": self.actions[act_index]}, self.is_nan, episode_id
