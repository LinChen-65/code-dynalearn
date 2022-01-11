#
import setproctitle
setproctitle.setproctitle("gnn-simu-vac@chenlin")

import os
import json
import sys
import pdb

sys.path.append("../sources")
from script import launch_scan


#name = "exp" #original
name = "safegraph-SanFrancisco"
#specs = json.load(open("./specs.json", "r"))["default"]
#specs = json.load(open(os.path.join(os.path.abspath('..'),"sources/specs.json"), "r"))["default"] #20211221
specs = json.load(open(os.path.join(os.path.abspath('..'),"sources/specs.json"), "r"))["safegraph"] #20211223
config = {
    "name": name,
    "path_to_covid": specs["path_to_data"],
    "epochs": 200, #original #20220109
    #"epochs": 3, #20220105
    #"epochs": 2, #test #20220110
    "type": ["rnn"],
    "model": [
        "DynamicsGATConv",
        #"FullyConnectedGNN", #original
        #"IndependentGNN", #original
        #"KapoorConv", #original
    ],
    "lag": [5],
    #"bias": [0.0, 0.25, 0.5, 0.75, 1.0], #original #20220109
    #"bias": [0.0], #20211228
    "bias": [1], #20220103
    #"val_fraction": 0.1, #original
    "val_fraction": 0.01, #20220105
    "gen_code": int(sys.argv[1]), #20220105, 控制读入的dataset (含义见gt_generator/wrap_data.py)
}
launch_scan(
    name,
    os.path.join(specs["path_to_data"], "covid"),
    #"../sources/run-covid.py", #path_to_script, 执行脚本的位置
    "../sources/run-covid-mine.py", #path_to_script, 执行脚本的位置 #20211223
    command=specs["command"],
    time="15:00:00",
    memory="8G",
    account=specs["account"],
    modules_to_load=specs["modules_to_load"],
    source_path=specs["source_path"],
    config=config,
    #devices=specs["devices"],
    device=specs["device"], #20211221
    verbose=2,
)
