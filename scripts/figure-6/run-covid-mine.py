import os
import json
import sys
import pdb

sys.path.append("../sources")
from script import launch_scan


name = "exp"
#specs = json.load(open("./specs.json", "r"))["default"]
#specs = json.load(open(os.path.join(os.path.abspath('..'),"sources/specs.json"), "r"))["default"] #20211221
specs = json.load(open(os.path.join(os.path.abspath('..'),"sources/specs.json"), "r"))["safegraph"] #20211223
config = {
    "name": name,
    "path_to_covid": specs["path_to_data"],
    "epochs": 200,
    "type": ["rnn"],
    "model": [
        "DynamicsGATConv",
        "FullyConnectedGNN",
        "IndependentGNN",
        "KapoorConv",
    ],
    "lag": [5],
    "bias": [0.0, 0.25, 0.5, 0.75, 1.0],
    "val_fraction": 0.1,
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
