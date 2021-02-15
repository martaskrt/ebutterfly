import os
import time
import json
import functools

import numpy as np
from skopt import Optimizer

def config_generator(original_config='config/vanilla_resnet_config.json',
                     parameters={'optimizer/weight_decay':\
                                    {'min':3,'max':9,'type':'float'},
                                 'optimizer/learning_rate':\
                                    {'min':1,'max':3,'type':'float'},
                                 'optimizer/lr_scheduler/step_size':\
                                    {'min':1,'max':8,'type':'int'},
                                 'optimizer/lr_scheduler/gamma':\
                                    {'min':1e-2,'max':5e-1,'type':'float'},
                                 'training/class_weights_alpha':\
                                    {'min':.1,'max':3,'type':'float'}},
                     max_iter=10, nb_GPU=4,
                     opt_config_dir='config/bayesian_search',
                     done_dir='config/bayesian_search/done'):
    """
    Generates config files of tests to run according to a bayesian optimization.

    Parameters
    ----------
    original_config: str
        Path to the original config json file.

    parameters: dict
        The keys are the name of the hyperparameters to optimize.
        The values are dictionnaries containing the min and max value, as well
        as the value type for each parameter (usually int or float).

    max_iter: int
        The maximal number of steps to optimize the hyperparameters.

    nb_GPU: int
        The parallelizability of the optimization. A batch of parameters will be
        created for each GPU to be runned parallely.

    opt_config_dir: str
        Path to the output directory of config in process.

    done_dir: str
        Path to the main directory in which to store the done files.
    """

    # First, make sure we have the config directory
    for main_dir in [opt_config_dir, done_dir]:
        try:
            os.mkdir(opt_config_dir)
            os.mkdir(done_dir)
        except:
            pass

    itr = 0
    while itr < max_iter:
        ### Get the config
        with open(original_config) as json_file:
            config = json.load(json_file)

        ### Prepare the optimizer to use
        opt = Optimizer(dimensions=[(parameters[param]['min'],
                                     parameters[param]['max']) \
                                        for param in parameters],
                acq_optimizer="sampling",
                n_random_starts=nb_GPU
            )

        ### Ask for the first set of parameters
        asked_params = opt.ask(n_points=nb_GPU)

        ### Run experiments according to this set on the gpus
        for test_index, test_values in enumerate(asked_params):
            for index, param in enumerate(parameters.keys()):
                param_value = test_values[index]
                param_value = param_value.astype(parameters[param]['type'])
                if type(param_value)==np.float64:
                    param_value = float(param_value)
                else:
                    param_value = int(param_value)
                functools.reduce(lambda X, Y: X[Y], [config] \
                    + param.split('/')[:-1])[param.split('/')[-1]] = param_value

            new_config_name = os.path.join(opt_config_dir,
                'config_itr_{}_SampledValues_{}.json'.format(itr,test_index))
            with open(new_config_name, 'w') as file:
                json.dump(config, file)

        print('Files edited, waiting for the results...')
        ### Wait for all the experiments to be finished
        best_scores = []
        for test_index, test_values in enumerate(asked_params):
            not_received = True
            while not_received:
                time.sleep(12)
                try:
                    with open(os.path.join(done_dir,
                        'config_itr_{}_SampledValues_{}.txt'.format(itr,
                                                           test_index))) as res:
                        best_scores.append(float(res.readline()))
                    not_received = False
                except:
                    continue
        ### Then, return the scores to the optimizer
        opt.tell(asked_params, best_scores)

        itr += 1

    ### Create a file to stop computation and then delete it
    with open(os.path.join(opt_config_dir, 'STOP'), 'w') as file:
        pass
    time.sleep(1)
    os.system('rm {}/STOP'.format(opt_config_dir))

    print('OPTIMIZATION FINISHED.')

if __name__ == '__main__':
    config_generator()
