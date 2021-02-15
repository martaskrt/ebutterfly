import os
import shutil
from lockfile import LockFile
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.train as train

def listener(opt_config_dir='config/bayesian_search',
             exec_dir='config/bayesian_search/in_progress',
             done_dir='config/bayesian_search/done',
             log_dir='runs/bayesian_search'):
    """
    Function that will be listening on a gpu. It waits for a file to be created
    and then works on it if it is available. Note the trick of moving the file
    in order to make sure there can't be 2 gpus working on the same file.

    Parameters
    ----------
    opt_config_dir: str
        Path to the directory containing the config files for the optimization.
        New generated config files should pop out in this directory.
    exec_dir: str
        Path to the directory containing the config files being executed.
    done_dir: str
        Path to the directory containing the config files finished. The skopt
        optimizer will fetch the results in here to go on.
    log_dir: str
        Path to the directory containing the logs of the runs executed.
    """
    for main_dir in [opt_config_dir, exec_dir, done_dir, log_dir]:
        try:
            os.mkdir(main_dir)
        except:
            pass

    print('Listening...')
    while True:
        received = False
        available_files = os.listdir(opt_config_dir)
        first_available_files = available_files.copy()
        if 'STOP' in first_available_files:
            break
        for wrong_file in first_available_files:
            if 'config' not in wrong_file:
                available_files.remove(wrong_file)

        if len(available_files)>0:
            config_file = available_files[0]
            try:
                lock = LockFile(opt_config_dir+'/'+config_file)
                with lock:
                    shutil.move(opt_config_dir+'/'+config_file, exec_dir+'/'+config_file)
                    # then it means that you've been able to move this file and thus it belongs to you.
                    received = True
            except:
                received = False

            if received==True:
                print('{} received. Running the experiment...'.format(config_file))
                best_scores = train.main(["--config_file", exec_dir+"/"+config_file, "--logdir", log_dir])#, "--debug"])
                with open(os.path.join(done_dir,config_file.split('.')[0]+'.txt'), 'w') as file:
                    file.write(str(best_scores['accuracy']))
                print('{} done. Heading to another experiment...'.format(config_file))


    print('Training Completed.')

if __name__ == '__main__':
    listener()
