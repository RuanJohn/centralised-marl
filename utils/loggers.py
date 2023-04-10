import os
import wandb
from typing import Optional

class WandbLogger: 

    def __init__(
            self,  
            exp_config: dict, 
            project_name: str = "centralised-marl-ppo-sweep",
            # TODO set this as an env variable
            # api_token = os.environ["WANDB_API_TOKEN"], 
            run_name: Optional[str] = None, 
    ) -> None:

        self._run = wandb.init(
            project = project_name, 
            name = run_name, 
            config = exp_config, 
            save_code = True, 
        )

    def write(self, logging_details: dict, step: Optional[int] =None): 
        
        if step is not None: 
            self._run.log(logging_details)
        else: 
            self._run.log(data=logging_details, step=step)

    def close(self): 

        self._run.finish()