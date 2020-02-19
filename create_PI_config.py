import json

data = {}
data['stable_gbw'] = 30000
data['control_zero'] = 7500
data['ssa_bw'] = 100000

with open("hobicat.json", "w") as write_file:
    json.dump(data, write_file)
