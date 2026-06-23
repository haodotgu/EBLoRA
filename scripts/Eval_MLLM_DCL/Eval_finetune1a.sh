#!/bin/bash


bash scripts/Eval_MLLM_DCL/eval_med.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task1.json configs/data_configs/MLLM-DCL/Med.json
bash scripts/Eval_MLLM_DCL/eval_ad.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task1.json configs/data_configs/MLLM-DCL/AD.json
bash scripts/Eval_MLLM_DCL/eval_sci.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task1.json configs/data_configs/MLLM-DCL/Sci.json
bash scripts/Eval_MLLM_DCL/eval_fin.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task1.json configs/data_configs/MLLM-DCL/Fin.json

bash scripts/Eval_MLLM_DCL/eval_ad.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task2.json configs/data_configs/MLLM-DCL/AD.json
bash scripts/Eval_MLLM_DCL/eval_sci.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task2.json configs/data_configs/MLLM-DCL/Sci.json
bash scripts/Eval_MLLM_DCL/eval_fin.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task2.json configs/data_configs/MLLM-DCL/Fin.json

bash scripts/Eval_MLLM_DCL/eval_sci.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task3.json configs/data_configs/MLLM-DCL/Sci.json
bash scripts/Eval_MLLM_DCL/eval_fin.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task3.json configs/data_configs/MLLM-DCL/Fin.json

bash scripts/Eval_MLLM_DCL/eval_fin.sh configs/model_configs/LLaVA/MLLM-DCL/eval/task4.json configs/data_configs/MLLM-DCL/Fin.json
