import re

FUTURENAV_R2R_RXR = {
    "annotation_path": "data/futurenav_r2r_rxr.json",
    "data_path": "",
    "tag": "3d"
}

data_dict = {
    "futurenav_r2r_rxr": FUTURENAV_R2R_RXR,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config["dataset_name"] = dataset_name
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["futurenav_r2r_rxr"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
