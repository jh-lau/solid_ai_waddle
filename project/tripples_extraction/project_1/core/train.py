import argparse

from .args import train_argparser, eval_argparser
from .config_reader import process_configs
from .spert import input_reader
from .spert.spert_trainer import SpERTTrainer


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, default='train',
                            help="Mode: 'train' or 'eval' or 'predict'")
    args, _ = arg_parser.parse_known_args()
    func_dict = {'train': _train, 'eval': _eval}

    try:
        func_dict[args.mode]()
    except KeyError:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python train.py train ...'")
