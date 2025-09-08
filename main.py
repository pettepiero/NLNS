""" Neural Large Neighborhood Search
    Copyright (C) 2020  André Hottung

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""

import logging
import numpy as np
import datetime
import os
import sys
import zipfile
import config
import train
import search
from actor import VrpActorModel
from critic import VrpCriticModel
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


VERSION = "0.3.0"

if __name__ == '__main__':
    run_id = np.random.randint(10000, 99999)
    config = config.get_config()

    # Creating output directories
    if config.output_path == "":
        config.output_path = os.getcwd()
    now = datetime.datetime.now()
    config.output_path = os.path.join(config.output_path, "runs", f"run_{now.day}.{now.month}.{now.year}_{run_id}")
    os.makedirs(os.path.join(config.output_path, "solutions"))
    os.makedirs(os.path.join(config.output_path, "models"))
    os.makedirs(os.path.join(config.output_path, "search"))
    if config.video:
        os.makedirs(os.path.join(config.output_path, "images"))

    # Create logger and log run parameters
    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Started")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("Version: {0}".format(VERSION))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")

    if config.mode == "train":
        actor = VrpActorModel(config.device, hidden_size=config.pointer_hidden_size).to(config.device)
        critic = VrpCriticModel(config.critic_hidden_size).to(config.device)

        model_path = train.train_nlns(actor, critic, run_id, config, load_dataset=False, save_dataset=True)
        search.evaluate_batch_search(config, model_path)

        if config.video:
            folder_path = os.path.join(config.output_path, "images") 
            output_path = folder_path.rstrip(os.sep) + ".zip"
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                with roots, dirs, files in os.walk(folder_path):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, folder_path)
                        zipf.write(abs_path, rel_path)
            print(f"\n****************************************************************")
            print(f"Zipped images in {output_path}")
            print(f"****************************************************************\n")

    elif config.mode == "eval_batch":
        if config.instance_path and not config.instance_path.endswith(".pkl"):
            raise Exception("Batch mode only supports .pkl instances files.")
        search.evaluate_batch_search(config, config.model_path)
    elif config.mode == "eval_single":
        search.evaluate_single_search(config, config.model_path, config.instance_path)
    elif config.mode == "multi_depot":
        search.evaluate_multi_depot_search(config, config.instance_path)
    else:
        raise Exception("Unknown mode")
