import os
import argparse
import multiprocessing
import yaml
import sys
import time
from Environment.utils.utils import parse_roadnet
import torch


def combine_include(d):
    if "includes" in d:
        full = {}
        for i in d["includes"]:
            i = combine_include(i)
            full.update(i)
        del d["includes"]
        full.update(d)
        d = full
    return d


def yaml_include(loader, node):
    # Get the path of the file to include
    file_path = node.value

    # Open and load the included YAML file
    with open(file_path, "r") as inputfile:
        return yaml.load(inputfile, Loader=yaml.Loader)


def parse_cityflow_config(
    args, configname, simulate_time_changable=False, cityflow_config_modify=""
):
    cf_conf = yaml.load(open(configname).read(), yaml.Loader)
    cf_conf = combine_include(cf_conf)
    cf_conf["ROADNET_INFO"], cf_conf["VIRTUAL_INTERSECTION_NAMES"] = parse_roadnet(
        cf_conf["ROADNET_FILE"]
    )
    # cf_conf['EPISODE_LEN'] = args.simulate_time * cf_conf['MIN_ACTION_TIME']
    if cf_conf["EPISODE_LEN"] % cf_conf["MIN_ACTION_TIME"] != 0:
        raise ValueError(
            "in cityflow config: EPISODE_LEN(%d) should be "
            "fully divided by MIN_ACTION_TIME(%d)"
            % (cf_conf["EPISODE_LEN"], cf_conf["MIN_ACTION_TIME"])
        )
    stcal = cf_conf["EPISODE_LEN"] // cf_conf["MIN_ACTION_TIME"]
    if args.simulate_time is None or args.simulate_time != stcal:
        if simulate_time_changable:
            if args.simulate_time is not None:
                print(
                    "[WARN ] simulate-time is set (%d), but will be "
                    "replaced by %d calculated from cityflow config."
                    % (args.simulate_time, stcal)
                )
            args.simulate_time = stcal
        else:
            print(args.simulate_time, stcal)
            raise ValueError(
                "in cityflow config: simulate-time isn't None, "
                "and is not same in cityflow config: %s" % configname
            )
    cf_conf["CONFIG_FILE"] = configname
    args.phase_skip_penalty = cf_conf["PHASE_SKIP"]
    args.green_time_penalty = cf_conf["GREEN_TIME"]
    args.green_skip_penalty = cf_conf["GREEN_SKIP"]
    if len(cityflow_config_modify) > 0:
        # modify config
        for mconfigs in cityflow_config_modify:
            for config in mconfigs.split(";"):
                keys, value = config.split("=")
                keys = keys.split(":")
                node = cf_conf
                for key in keys[:-1]:
                    node = node[key]
                if keys[-1] in node.keys():
                    # for bool type, do specially
                    if isinstance(node[keys[-1]], bool):
                        node[keys[-1]] = value != "0" and value.lower() != "false"
                    else:
                        node[keys[-1]] = type(node[keys[-1]])(value)
                else:
                    node[keys[-1]] = value
    return cf_conf


def parse_args(input_args=sys.argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-folder", type=str, default="logs/")
    parser.add_argument("--work-folder", type=str, default="./")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--use_cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--wrapper-model", type=str, default="")
    # st_default = int(1e9) + random.random()
    parser.add_argument("--simulate-time", type=int, default=None)
    parser.add_argument("--cityflow-config", type=str, default="")
    parser.add_argument(
        "--cityflow-config-modify",
        type=str,
        action="append",
        default=[],
        help="change cityflow configs manually, "
        "write as `key1:subkey1=val1;key2=val2`. support "
        "calling multiple times.",
    )
    parser.add_argument("--cityflow-log", type=str, default="")
    parser.add_argument("--preload-model-file", type=str, default="")
    parser.add_argument(
        "--test-round",
        type=int,
        default=0,
        help="when set positive integer, start evaluate mode and " "show result.",
    )
    parser.add_argument("--mode", type=str, default="train")

    parser.add_argument("--fixed-time", type=float, default=30)

    parser.add_argument("--NM-lane-embedding-size", type=int, default=32)
    parser.add_argument("--NM-road-predict-hidden", type=int, default=32)
    parser.add_argument("--NM-scale-by-lane-number", action="store_true", default=False)
    parser.add_argument("--NM-phase-loss-weight", type=float, default=0)
    parser.add_argument("--NM-volume-loss-weight", type=float, default=0)
    parser.add_argument(
        "--NM-phase-loss-with-replay", action="store_true", default=False
    )
    parser.add_argument(
        "--constraint",
        type=str,
        choices=["PhaseSkip", "GreenTime", "GreenSkip", "None", "All"],
        default="None",
    )
    parser.add_argument("--CM-selected-inner", type=str, default="")

    parser.add_argument("--gammas", type=str, default="0.99")

    parser.add_argument("--multi-agent", type=str, default="")

    parser.add_argument("--enable-wandb", action="store_true", default=False)
    parser.add_argument("--wandb-entity-name", type=str, default="")
    parser.add_argument("--wandb-api-key", type=str, default="")
    parser.add_argument("--wandb-project-name", type=str, default="ctraffic")
    parser.add_argument("--wandb-sync-mode", type=str, default="online")

    # common args
    parser.add_argument("-m", "--main", type=str)
    parser.add_argument("-a", "--agent", type=str)
    parser.add_argument("--env", type=str, default="ctraffic")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("-n", "--n-frames", type=int, default=1000000)
    parser.add_argument("-t", "--target-reward", type=float, default=1e100)
    parser.add_argument("-tx", "--tensorboardx-comment", type=str, default="")
    parser.add_argument("-s", "--model-save-path", type=str, default="")
    parser.add_argument("-si", "--save-interval", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("-r", "--render-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("-li", "--log-interval", type=int, default=100)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("-c", "--config", type=str, action="append", default=[])
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--evaluate-round", type=int, default=10)
    parser.add_argument("--evaluate-interval", type=int, default=10)
    parser.add_argument("--feature-hidden-size", type=int, default=512)
    parser.add_argument("-g", "--gpu-to-use", type=str, default="")
    parser.add_argument("-dev", "--development", action="store_true", default=False)
    parser.add_argument("--ablation", type=str, default="", choices=["", "fixed_lambda", "advantage_lambda_update"])
    parser.add_argument("--results_path", type=str, default="./results", help="path to save results")
    parser.add_argument("--use")

    args = parser.parse_args(input_args[1:])
    yaml.add_constructor("!include", yaml_include, Loader=yaml.Loader)

    # if config file is set, read it and parse it before cmd arguments
    if len(args.config) > 0:
        confs = {}
        for config in args.config:
            one_conf = yaml.load(open(config), Loader=yaml.Loader)
            confs.update(combine_include(one_conf))
        carg = []
        for conf in confs.keys():
            carg.append("--" + conf)
            if not isinstance(confs[conf], bool):
                carg.append(str(confs[conf]))
            else:
                if not confs[conf]:
                    carg = carg[:-1]
        # print('config:', carg, confs, carg + sys.argv[1:])
        args = parser.parse_args(carg + input_args[1:])

    # if thread number not set, use half cpus
    if args.threads == -1:
        cpu_count = multiprocessing.cpu_count() // 2
        if cpu_count < 1:
            cpu_count = 1
        args.threads = cpu_count
        print("threads unset, auto use %d threads" % args.threads)

    if args.seed == -1:
        args.seed = int(time.time() * 1000000000) % (2**31)

    if args.tensorboardx_comment == "-":
        args.tensorboardx_comment = ""

    # CityFlow args processing
    tcc = args.cityflow_config
    args.cityflow_config = parse_cityflow_config(
        args, args.cityflow_config, True, args.cityflow_config_modify
    )
    tcc = tcc.split(",")
    for i in range(len(tcc)):
        tcc[i] = parse_cityflow_config(
            args, tcc[i], cityflow_config_modify=args.cityflow_config_modify
        )
    while args.threads > len(tcc):
        tcc += tcc
    tcc = tcc[: args.threads]
    args.train_cityflow_config = tcc

    return args


if __name__ == "__main__":
    import sys

    print(sys.argv)
    print(parse_args())
