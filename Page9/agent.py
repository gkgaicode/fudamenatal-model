import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FoundationalAIArchitectAgent:
    def __init__(self, config_path="agent_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        logging.info(f"Initialized Agent: {self.config['agent_name']} v{self.config['version']}")

    def execute_lifecycle_stage(self, stage_name, **kwargs):
        """Orchestrates the active skill modules based on execution targets."""
        logging.info(f"Initiating Execution Loop: Stage [{stage_name}]")
        
        if stage_name == "DATA_PREP":
            import data_prep_skill
            data_prep_skill.run(kwargs.get("text_files"), kwargs.get("vocab_size", 32000))
            
        elif stage_name == "PRETRAIN_OR_SFT":
            # Triggers distributed cluster run commands
            num_gpus = kwargs.get("num_gpus", 1)
            config_file = kwargs.get("ds_config", "ds_config_zero3.json")
            script = "sft_train_run.py" if kwargs.get("is_sft") else "pretrain_run.py"
            cmd = f"deepspeed --num_gpus={num_gpus} {script} --config {config_file}"
            logging.info(f"Dispatching cluster command: {cmd}")
            os.system(cmd)
            
        elif stage_name == "ALIGNMENT_DPO":
            num_gpus = kwargs.get("num_gpus", 1)
            cmd = f"deepspeed --num_gpus={num_gpus} dpo_train_run.py"
            logging.info(f"Dispatching human preference loop: {cmd}")
            os.system(cmd)
            
        elif stage_name == "EXPORT_AND_QUANTIZE":
            import convert_and_quantize_skill
            convert_and_quantize_skill.run(
                kwargs.get("ds_checkpoint"), 
                kwargs.get("hf_out_dir"), 
                kwargs.get("awq_out_dir")
            )
            
        elif stage_name == "SERVE":
            model_path = kwargs.get("model_path")
            port = kwargs.get("port", 8000)
            cmd = f"python3 -m vllm.entrypoints.openai.api_server --model {model_path} --quantization awq --port {port} --host 0.0.0.0"
            logging.info(f"Spinning up PagedAttention API engine: {cmd}")
            os.system(cmd)
            
        else:
            logging.error(f"Stage {stage_name} undefined in agent skill manifest.")

if __name__ == "__main__":
    agent = FoundationalAIArchitectAgent()
    # Example command execution mapping context:
    # agent.execute_lifecycle_stage("DATA_PREP", text_files=["corpus.txt"])
