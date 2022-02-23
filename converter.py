import json
import jsonlines 
import gzip 
import os.path as osp 
import h5py 
from tqdm import tqdm 
import numpy as np 
import ipdb

def connect(ends):
    d0, d1 = np.abs(np.diff(ends, axis=0))[0]
    if d0 > d1: 
        return np.c_[np.linspace(ends[0, 0], ends[1, 0], d0+1, dtype=np.int32),
                     np.round(np.linspace(ends[0, 1], ends[1, 1], d0+1))
                     .astype(np.int32)]
    else:
        return np.c_[np.round(np.linspace(ends[0, 0], ends[1, 0], d1+1))
                     .astype(np.int32),
                     np.linspace(ends[0, 1], ends[1, 1], d1+1, dtype=np.int32)]

def grid_to_loc(grid_x, grid_y, grid_resolution, lower_bound, upper_bound):
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    real_word_x = grid_x * grid_size[0] + lower_bound[2]
    real_word_y = grid_y * grid_size[1] + lower_bound[0]
    return real_word_x, real_word_y


def get_data_from_json(path_root,split):
    path = path_root.format(split=split)
    with gzip.open(path) as f:
        data = json.load(f)
    return data


def get_data_from_results(path):
    data = {}
    with open(path,"r+") as f:
        for items in jsonlines.Reader(f):
            data[items["episode_id"]] = items
    return data

def get_map_info(scene_name, root_path):
    gmap_path = osp.join(root_path, f"{scene_name}_gmap.h5")
    with h5py.File(gmap_path, "r") as f:
        nav_map  = f['nav_map'][()]
        bounds = f['bounds'][()]
     
    grid_dimensions = (nav_map.shape[0], nav_map.shape[1]) # row, column 
    return grid_dimensions, bounds

def grid_to_sim_loc(grid_x, grid_y, grid_resolution, lower_bound, upper_bound):
    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    real_word_x = grid_x * grid_size[0] + lower_bound[2]
    real_word_y = grid_y * grid_size[1] + lower_bound[0]
    return real_word_x, real_word_y


def path_loc_to_sim_loc(grid_path, grid_resolution, lower_bound, upper_bound):
    sim_path = []

    for point in grid_path:
        p1, p2 ,p3 = point 
        sim_x, sim_y = grid_to_sim_loc(p1,p2,grid_resolution, lower_bound, upper_bound)
        sim_path.append([sim_y, p3, sim_x])
    
    return sim_path 

def convert_and_save(results, our_dir,map_root):
    data_dict = {}
    for k,v in tqdm(results.items()):
        data_dict[k] = v
        grid_resolution, bounds = get_map_info(v["scene_name"], map_root)
        upper_bound, lower_bound = bounds[0], bounds[1]
        ## interpolate path 
        
        end_point  = [int(v["path"][-1][:2][0]),int(v["path"][-1][:2][1])]
        sec_point = [int(v["path"][-2][:2][0]),int(v["path"][-2][:2][1])]
        inter_points = connect(np.array([sec_point,
                                         end_point]))
  
        inter_points = [[p[0],p[1],v["path"][-1][2]] for p in inter_points]
     
        path = v["path"][:-2]
        path.extend(inter_points)
        assert len(path) >= len(v["path"])
        sim_loc_path = path_loc_to_sim_loc(path,grid_resolution, lower_bound, upper_bound)
        data_dict[k]["sim_path"] = sim_loc_path  
    with open(our_dir, "w") as f:
        json.dump(data_dict,f)
    print("DONE !!")


if __name__ == "__main__":
    #val_seen_data_path = "data/mln_v1/annt/val_seen/val_seen.json.gz"
    #val_seen_result_path = "data/MLN-CE_results/val_seen_results_with_path.jsonl"

    #val_unseen_path = "data/mln_v1/annt/val_unseen/val_unseen.json.gz"
    val_unseen_result_path = "data/MLN-CE_results/val_unseen_results_with_path.jsonl"

    h5_map_path = "data/mln_v1/maps/gmap_floor1_mpp_word_0.05_channel_last_with_bounds"
    
    #val_seen_data = get_data_from_json(val_seen_data_path)
    #val_seen_result = get_data_from_results(val_seen_result_path)

    #val_unseen_data = get_data_from_json(val_unseen_data_path)
    val_unseen_result = get_data_from_results(val_unseen_result_path)

   #convert_and_save(val_seen_result, "best_glove_val_seen_results.json", h5_map_path)
    convert_and_save(val_unseen_result, "best_glove_val_unseen_insert_results.json", h5_map_path)

    



