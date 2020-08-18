import cloudpickle
from pathlib import Path

class BestAndLatestModelCheckpoint(object):
    def __init__(self, save_directory, best_name="best.pkl", latest_name="latest.pkl"):
        self.best_value = 10**9
        self.save_directory= Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.best_path = self.save_directory / best_name
        self.latest_path = self.save_directory / latest_name


    def __call__(self, pred, model):
        
        if pred < self.best_value:
            self.best_value = pred

            with open(self.best_path, "wb") as f:
                #print("Update best weight! loss : {}".format(self.best_value), flush=True)
                cloudpickle.dump(model, f)

        with open(self.latest_path, "wb") as f:
            
            cloudpickle.dump(model, f)



