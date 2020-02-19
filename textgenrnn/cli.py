import argparse
import datetime
import sys

from .textgenrnn import textgenrnn, textgenrnn_texts_from_file_context, textgenrnn_texts_from_file


def make_parser():
    default_name = 'textgenrnn-cli-%s' % datetime.datetime.now().isoformat().replace(':', '-')
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', '-f')
    ap.add_argument('--load-model')
    ap.add_argument('--num-cycles', '-n', type=int, default=100)
    ap.add_argument('--epochs-per-cycle', type=int, default=10)
    ap.add_argument('--generations-per-cycle', type=int, default=5)
    ap.add_argument('--max-gen-length', default=400, type=int)
    ap.add_argument('--name', default=default_name)
    ap.add_argument('--context', default=False, action='store_true')
    ap.add_argument('--single-text', default=False, action='store_true')
    for key, value in textgenrnn.default_config.items():
        if key == 'single_text':
            continue
        ap.add_argument(
            '--%s' % key.replace('_', '-'),
            dest='_config_%s' % key,
            type=type(value),
            default=value,
            metavar='VALUE',
            help='default: %s' % value,
        )
    return ap


def cli():
    ap = make_parser()
    args = ap.parse_args()
    # HACK: since we can't modify the config without a JSON file, we need to modify the class-level config instead...
    for key, value in vars(args).items():
        if key.startswith('_config_'):
            key = key[8:]
            assert key in textgenrnn.config
            textgenrnn.config[key] = value

    tg = textgenrnn(name=args.name)

    if args.load_model:
        tg.load(args.load_model)

    context_labels = None
    if args.context:
        texts, context_labels = textgenrnn_texts_from_file_context(args.input_file)
    elif args.single_text:
        with open(args.input_file, 'r', encoding='utf8', errors='ignore') as f:
            texts = [f.read()]
    else:
        texts = textgenrnn_texts_from_file(args.input_file, header=False)

    for cycle in range(1, args.num_cycles + 1):
        print('# Cycle %d/%d' % (cycle, args.num_cycles), file=sys.stderr)
        tg.train_on_texts(
            texts=texts,
            context_labels=context_labels,
            num_epochs=args.epochs_per_cycle,
            gen_epochs=0,
            single_text=args.single_text,
            save_epochs=0,
        )

        tg.generate_samples(
            n=args.generations_per_cycle,
            max_gen_length=args.max_gen_length,
        )

        checkpoint_file = '%s_%s.hdf5' % (args.name, cycle)
        print('=>> Saving to %s' % checkpoint_file)
        tg.save(checkpoint_file)
