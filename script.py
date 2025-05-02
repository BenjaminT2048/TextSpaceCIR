import glob
import json

paths = glob.glob("configs/circo*.json")
for path in paths:
    file = json.load(open(path))
    file['dataset'] = 'cirr'
    file['database_path']  = "CIRR/documents_pool_p095_t1.json"
    file['query_path'] = "CIRR/CIRR_query_diversified_temp2.json"
    if file['top_k'] == 5:
        file['top_k'] = 1
        path = path.replace("map5", "map1")
    if file['top_k'] == 10:
        file['top_k'] = 5
        path = path.replace("map10", "map5")
    if file['top_k'] == 25:
        file['top_k'] = 10
        path = path.replace("map25", "map10")
    path = path.replace("circo", "cirr")
    json.dump(file, open(path, "w"))