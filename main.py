import ai

from util import build_model, get_filter, build_dataset
from model import hardcoded as hardcoded_model


class CLI:
    '''Command line interface.

    A command is a method of this class.
    Arguments can be either arguments of that method or the class as a whole.

    HELP: python main.py

    USAGE: python main.py <command> [arguments]

    EXAMPLE: python main.py train --name='general' --kernel='random'
    '''

    def __init__(s,
        kernel='sobel',
        dataset='cifar10',
        device='cuda',
        val_batch_size=256,
        single_batch=False,
        n_samples=4,
    ):
        # a study is a basically a folder for storing all the results
        # path: $AI_LAB_PATH/filter/<kernel>/<dataset>
        s._study = ai.Study(f'filter/{kernel}/{dataset}')

        # data
        ds = build_dataset(dataset, get_filter(kernel))
        s._train_ds, s._val_ds  = ds.split(.99, .01)
        s._val_iter = s._val_ds.iterator(val_batch_size, device, train=False, single_batch=single_batch)
        s._samples = s._val_ds.sample(n_samples, device)

        s._device = device
        s._single_batch = single_batch

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # COMMANDS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def clean(s):
        s._study.clean() # delete everything in the study

    def examine_data(s):
        # create an image grid of data samples
        # column 1: original images
        # column 2: post-filter images
        ai.util.save_img_grid(
            s._study.path / 'data.png',
            [s._samples['x'], s._samples['y']],
        )

    def compare_greyscale(s):
        # check whether the greyscale conversion should be learned
        s._compare(
            'compare_greyscale',
            ('basic', hardcoded_model(False)),
            ('learned', hardcoded_model(True)),
        )

    def train(s, name='simplest/test', steplimit=10_000, **hparams):
        trial = s._create_trial(name) # a trial is a single training run
        model = build_model(name.split('/')[0])
        s._train(model, trial.hook(), steplimit=steplimit, **hparams)

    def hparam_search(s, name='simplest/search', n_trials=16, clean=False, prune=False, steplimit=5_000):
        model = build_model(name.split('/')[0])

        # create an experiment (a collection of trials)
        exp = s._study.experiment(name, clean=clean, prune=prune, val_data=s._val_iter)

        # run a hyperparam search
        exp.run(n_trials, lambda trial: s._train(
            model,
            trial.hook(),

            # both specifies the searchable hyperparameter space for the whole experiment
            # and selects the exact hparams for this specific trial
            trial.hp.lst('opt', ['sgd', 'adam', 'adamw']),
            trial.hp.log('lr', 1e-6, 1e-1),
            trial.hp.lst('grad_clip', [False, True]),

            steplimit=steplimit,
        ))

        print(exp.best_hparams)

    def train_best(s, name='simplest/best', search='simplest/search', steplimit=10_000):
        # load best hyperparameters from a search
        hparams = s._study.experiment(search).best_hparams

        trial = s._create_trial(name)
        model = build_model(name.split('/')[0])
        s._train(model, trial.hook(), steplimit=steplimit, **hparams)

    def inspect_model(s, name='simplest/best'):
        model = build_model(name.split('/')[0])
        model.init(s._study.trial(name).model_path()) # load weights from disk
        for k, v in model.state_dict().items():
            print(f'{k}\n{v}\n')

    def count_params(s, name='lightest'):
        model_name = name.split('/')[0]
        print('model name:', model_name)
        print('number of params:', build_model(model_name).n_params)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # HELPERS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compare(s, name, *models, **train_kw):
        losses = {}
        imgs = [s._samples['x'], s._samples['y']]
        for label, model in models:
            trial = s._create_trial(f'{name}/{label}')
            losses[label] = s._train(model, trial.hook(), **train_kw)
            imgs.append(model(s._samples['x']))

        print('\n~~~ FINAL VAL LOSSES ~~~')
        for label, _ in models:
            print(f'{label}: {losses[label]:.4f}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~')

        ai.util.save_img_grid(s._study.path / name / 'comparison.png', imgs)

    def _create_trial(s, name, save_snapshots=True):
        return s._study.trial(
            name,
            clean=True,
            save_snapshots=save_snapshots,
            val_data=s._val_iter,
            sampler=lambda path, step, model: ai.util.save_img_grid(
                path / f'{step}.png',
                [s._samples['x'], s._samples['y'], model(s._samples['x'])],
            ),
        )

    def _train(s, model, hook, opt='sgd', lr=1e-2, grad_clip=False, batch_size=64, steplimit=10_000):
        trainer = ai.Trainer(
            ai.train.Target(), # training environment
            s._train_ds.iterator(batch_size, s._device, train=True, single_batch=s._single_batch),
        )

        trainer.train(
            model.init().to(s._device),
            ai.opt.from_name(opt, model, lr=lr, grad_clip=grad_clip), # optimizer
            hook,
            steplimit=steplimit,
        )

        return trainer.validate(model, s._val_iter)


if __name__ == '__main__':
    ai.run(CLI)
