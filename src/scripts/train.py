import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='../config', config_name='train')
def main(conf: DictConfig):
    conf = OmegaConf.to_container(conf) 
    # train(conf)

if __name__ == '__main__':
    main()