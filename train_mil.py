import argparse
import warnings
import os
import subprocess
from pathlib import Path
warnings.filterwarnings('ignore')

OFFICIAL_MULTISCALE = {
    'CS_MIL': 'csmil',
    'DAS_MIL': 'dasmil',
    'H2_MIL': 'h2_mil',
    'HAG_MIL': 'hag_mil',
}


def run_official_multiscale(config, options):
    """Use the existing audited research runner for the four scale-fusion methods."""
    import yaml

    for option in options or []:
        key, separator, value = option.partition('=')
        if not separator or key not in {
            'General.seed', 'General.device', 'Dataset.cohorts',
            'Dataset.folds', 'Model.arms',
        }:
            raise ValueError(f'unsupported multiscale option: {option}')
        section, name = key.split('.')
        config[section][name] = yaml.safe_load(value)

    method = OFFICIAL_MULTISCALE[config['General']['MODEL_NAME']]
    runner = Path(config['Model']['research_runner']).resolve()
    interpreter = Path(config['Model']['research_python']).resolve()
    if not runner.is_file():
        raise FileNotFoundError(f'research runner not found: {runner}')
    if not interpreter.is_file():
        raise FileNotFoundError(f'research Python not found: {interpreter}')
    cohorts = config['Dataset']['cohorts']
    folds = config['Dataset']['folds']
    arms = config['Model']['arms']
    if (not isinstance(cohorts, list) or not isinstance(folds, list)
            or not isinstance(arms, list) or not cohorts or not folds or not arms):
        raise ValueError('cohorts, folds, and arms must be non-empty lists')
    command = [
        str(interpreter), str(runner), '--stage', 'unified-fits', '--methods', method,
        '--cohorts', *cohorts, '--folds', *map(str, folds),
        '--seeds', str(config['General']['seed']), '--arms', *arms,
        '--output-root', str(Path(config['Logs']['output_root']).resolve()),
        '--device', f"cuda:{config['General']['device']}",
    ]
    subprocess.run(command, check=True, cwd=runner.parents[1])


def main(arg):
    yaml_path = arg.yaml_path
    print(f"MIL-yaml path: {yaml_path}")
    import yaml
    with open(yaml_path, encoding='utf-8') as handle:
        raw_config = yaml.safe_load(handle)
    if raw_config['General']['MODEL_NAME'] in OFFICIAL_MULTISCALE:
        run_official_multiscale(raw_config, arg.options)
        return

    from utils.yaml_utils import read_yaml,update_config_from_options
    from process.process_all import process
    from utils.general_utils import get_time,merge_k_fold_logs
    args = read_yaml(yaml_path)
    # dinamically update the config file with the options
    if arg.options:
        args = update_config_from_options(args,arg.options)
    
    if args.Dataset.dataset_root_dir == {} and args.Dataset.dataset_csv_path != None:
        '''
        None-fold split
        '''
        log_root_dir = args.Logs.log_root_dir
        os.makedirs(log_root_dir,exist_ok=True)
        sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
        os.makedirs(sub_dir,exist_ok=True)
        args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{get_time()}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}')
        process(args,yaml_path,arg.options)

    else:
        '''
        k-fold split
        '''
        dataset_root_dir = args.Dataset.dataset_root_dir
        k_fold_csv_paths = sorted([os.path.join(dataset_root_dir,path) for path in os.listdir(dataset_root_dir)])
        process_time = get_time()
        for k_idx,k_fold_csv_path in enumerate(k_fold_csv_paths):
            args.Dataset.dataset_csv_path = k_fold_csv_path
            now_fold = k_idx+1
            args.Dataset.now_fold = now_fold
            log_root_dir = args.Logs.log_root_dir
            os.makedirs(log_root_dir,exist_ok=True)
            sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
            os.makedirs(sub_dir,exist_ok=True)
            if now_fold != None:
                fold_dir = f'fold_{now_fold}'
                args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}/{fold_dir}')
            os.makedirs(args.Logs.now_log_dir,exist_ok=True)
            process(args,yaml_path,arg.options)
            print(f'K-Fold:{k_idx+1} Done!')
        fold_total_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}')
        merge_k_fold_logs(fold_total_log_dir,args.General.process_pipeline)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/path/to/your/yaml',help='path to MIL-yaml file')
    parser.add_argument('--options',nargs='+',help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the yaml config file')
    arg = parser.parse_args()
    main(arg)
