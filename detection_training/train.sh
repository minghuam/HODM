#!/bin/bash
python fcn32_obj_solve.py training_data.txt model/OBJ 4000 1 2>&1 | tee fcn32_obj_solve.log
