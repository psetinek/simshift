import os
import sys
import traceback
from datetime import datetime

import hydra
import petname
from omegaconf import DictConfig, OmegaConf

from simshift.run import run
from simshift.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    set_seed(cfg.seed)

    print("#" * 88, "\nStarting with configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 88, "\n")

    if not os.path.exists(cfg.output_path):
        dict_cfg = OmegaConf.to_container(cfg)
        os.makedirs(dict_cfg["output_path"], exist_ok=True)
        dict_cfg["ckp_dir"] = dict_cfg["output_path"]
        cfg = OmegaConf.create(dict_cfg)

    try:
        if cfg.logging.run_id is None:
            date_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
            random_petname = petname.generate(2, separator="_")
            cfg.logging.run_id = f"{random_petname}_{date_and_time}"
        # start with config
        run(cfg)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
