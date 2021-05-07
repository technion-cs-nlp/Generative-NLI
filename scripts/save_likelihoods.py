import torch, argparse, sys

def save_likelihoods(model_path, data_dir_prefix='./data/snli_1.0/cl_snli', output_file=None):
    pass
def parse_cli():
    p = argparse.ArgumentParser(description='Save likelihoods for a given model and dataset')
    p.set_defaults(subcmd_fn=save_likelihoods)
    p.add_argument('--model-path', help='Path to your model', required=True)
    p.add_argument('--data-dir-prefix', help='Path to the dataset', default='./data/snli_1.0/cl_snli')
    p.add_argument('--output-file', '-o', help='Where would you like to save the results', default=None)

    # sp = p.add_subparsers(help='Sub-commands')

    # # Experiment config
    # sp_exp = sp.add_parser('train', help='Train a model')
    # sp_exp.set_defaults(subcmd_fn=save_likelihoods)
    # sp_exp.add_argument('--run-name', '-n', type=str,
                        # help='Name of run and output file', required=True)

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    if torch.cuda.device_count() > 1:
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))