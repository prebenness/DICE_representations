import os
import torch
import torchattacks
import xlwt
from torch.nn import DataParallel
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from representations import representations
from utils.parse_args import parse_args
from utils.print_easydict import print_easydict
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.config import cfg
from utils.model_sl import load_model, save_model
from utils.dummy_attack import DummyAttack

args = parse_args('Beginning.')

from util_func import StatLogger, set_seed, make_model, get_data
from train import train, train_causal_poison, train_causal_adv, train_causal_attack, train_adv, eval_adv

def main(model, device, train_loader, optimizer, scheduler, tfboardwriter, loss_logger, acc_logger):
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        model.train()
        print('Starting training')
        if cfg.TRAIN.MODE == 'causal_poison':
            lossdict, accdict = train_causal_poison(model, device, train_loader, optimizer[0], optimizer[1], epoch, tfboardwriter)
        elif cfg.TRAIN.MODE == 'causal_adv':
            lossdict, accdict = train_causal_adv(model, device, train_loader, optimizer[0], optimizer[1], epoch, tfboardwriter)
        elif cfg.TRAIN.MODE == 'causal_attack':
            lossdict, accdict = train_causal_attack(model, device, train_loader, optimizer[0], optimizer[1], epoch, tfboardwriter)
        elif cfg.TRAIN.MODE == 'adv':
            lossdict, accdict = train_adv(model, device, train_loader, optimizer, epoch, tfboardwriter)
        elif cfg.TRAIN.MODE == 'baseline':
            lossdict, accdict = train(model, device, train_loader, optimizer, epoch, tfboardwriter)
        else:
            raise ValueError('unrecognized arguments for train mode.')

        if isinstance(scheduler, list):
            for sche in scheduler:
                sche.step()
        else:
            scheduler.step()

        # log current epoch-level statistics
        loss_logger.add(lossdict)
        acc_logger.add(accdict)

        # save checkpoints
        if epoch >= cfg.SAVE_EPOCH and epoch % cfg.SAVE_FREQ == 0:
            print(f'Evaluating model at epoch {epoch}')
            evaluation(model)

            save_model(model, os.path.join(cfg.SAVE_PATH, 'epoch{}.pt'.format(epoch)))
            if isinstance(optimizer, list):
                opt_names = ['causal', 'conf']
                for name, opt in zip(opt_names, optimizer):
                    torch.save(opt.state_dict(),
                                os.path.join(cfg.SAVE_PATH, '{}_opt_epoch{}.tar'.format(name, epoch)))
            else:
                torch.save(optimizer.state_dict(),
                            os.path.join(cfg.SAVE_PATH, 'opt_epoch{}.tar'.format(epoch)))

        #sample and save image from conf bank
        if 'causal' in cfg.TRAIN.MODE and  (epoch == 1  or epoch % cfg.SAVE_FREQ == 0):
            banksample_path = os.path.join(cfg.OUTPUT_PATH, f'confbank_sample')
            if not os.path.exists(banksample_path):
                os.mkdir(banksample_path)
            banksample_path = os.path.join(banksample_path, f"epoch{epoch}")
            if not os.path.exists(banksample_path):
                os.mkdir(banksample_path)
            if isinstance(model, DataParallel):
                model.module.erb.sample_and_save(banksample_path, num=10)
            else:
                model.erb.sample_and_save(banksample_path, num=10)

    # post-training
    loss_logger.save()
    acc_logger.save()


def evaluation(model=None):
    loss_dict, acc_dict = eval_adv(model, device, test_loader, 0, tfboardwriter, 'Test', att=cfg.ATTACK.LOSS_TYPE)


def test_robustness(model, data_loader, device):
    attack_factories = {
        'dummy_attacker': DummyAttack,
        'pgd20_linf': lambda m: torchattacks.PGD(m, eps=8/255, alpha=2/255, steps=20, random_start=True),
        'pgd40_linf': lambda m: torchattacks.PGD(m, eps=8/255, alpha=4/255, steps=40, random_start=True),
        'pgd20_l2': lambda m: torchattacks.PGDL2(m, eps=1.0, alpha=0.2, steps=20, random_start=True),
        'pgd40_l2': lambda m: torchattacks.PGDL2(m, eps=1.0, alpha=0.2, steps=40, random_start=True),
        'fgsm_linf': lambda m: torchattacks.FGSM(m, eps=8/255),
        'cw20_l2': lambda m: torchattacks.CW(m, c=1, kappa=0, steps=20),
        'cw40_l2': lambda m: torchattacks.CW(m, c=1, kappa=0, steps=20), 
    }

    results = {}
    for attack_name, attack_factory in attack_factories.items():
        print(f'Testing on {attack_name}')
        clean_acc, adv_acc = test_attack(
            model, attack_factory=attack_factory, test_loader=data_loader, device=device
        )
        results[attack_name] = (clean_acc, adv_acc)

    model_dir, model_name = os.path.split(model_path)
    log_name = '.'.join([model_name.split('.')[0], 'txt'])
    log_file = os.path.join(model_dir, log_name)

    res_text = '\n'.join(
        [ f'{attack_name} clean acc: {v[0]:10.8f} adv acc: {v[1]:10.8f}' for attack_name, v in results.items() ]
    )
    with open(log_file, 'w') as w:
        w.write(res_text)

    print(res_text)


def test_attack(model, attack_factory, test_loader, device):
    num_corr, num_corr_adv, num_tot = 0, 0, 0
    for (x, y) in test_loader:
        x = x.to(device)
        y = y.to(device)

        # Get clean test preds
        y_pred = model(x)

        # Craft adversarial samples
        attacker = attack_factory(model)
        x_adv = attacker(x, y)
        y_pred_adv = model(x_adv)

        num_corr += (y_pred.argmax(dim=1) == y).sum().item()
        num_corr_adv += (y_pred_adv.argmax(dim=1) == y).sum().item()
        num_tot += y.shape[0]

        print(f'Processed {num_tot:5d} of {len(test_loader.dataset):5d} samples, current tally: Clean acc: {num_corr / num_tot:4.3f} Adv acc: {num_corr_adv / num_tot:4.3f}')

    # Compute and store results
    clean_acc = num_corr / num_tot
    adv_acc = num_corr_adv / num_tot

    res_text = f'Test completed: Clean acc: {clean_acc} Adv acc: {adv_acc}'
    print(res_text)

    return clean_acc, adv_acc


if __name__ == '__main__':
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wb = xlwt.Workbook()
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))
    wb.__save_path = str(Path(cfg.OUTPUT_PATH) / (now_time + '.xls'))
    loss_logger = StatLogger()
    acc_logger = StatLogger()

    set_seed()
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    train_loader, val_loader, test_loader = get_data()

    model = make_model(cfg.ARCH)
    model = model.cuda()

    if cfg.ARCH == 'causal_v0':
        causal_opt = optim.SGD(params=model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
        conf_opt = optim.SGD(params=model.conf_mlp.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
        optimizers = [causal_opt, conf_opt]
    else:
        optimizers = optim.SGD(params=model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    optimizer = optimizers
    model = torch.nn.DataParallel(model)

    model_path, optim_path = '', ''
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path, strict=False)

    if cfg.lr_mode == 'cyclic':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_mode == 'piecewise':
        if isinstance(optimizer, list):
            scheduler = []
            for opt in optimizer:
                sche = optim.lr_scheduler.MultiStepLR(opt,
                                                        milestones=cfg.TRAIN.LR_STEP,
                                                        gamma=cfg.TRAIN.LR_DECAY,
                                                        last_epoch=cfg.TRAIN.START_EPOCH - 1)
                scheduler.append(sche)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=cfg.TRAIN.LR_STEP,
                                                        gamma=cfg.TRAIN.LR_DECAY,
                                                        last_epoch=cfg.TRAIN.START_EPOCH - 1,
                                                        verbose=cfg.TRAIN.VERBOSE)

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / (now_time + '.log'))) as _:
        print_easydict(cfg)
        if cfg.EVAL:
            test_robustness(model, test_loader, device=device)
            evaluation(model)
            
        elif cfg.REPR:
            representations(model, model_path, train_loader, test_loader)
        else:
            main(model, device, train_loader,
                optimizer, scheduler, tfboardwriter,
                loss_logger, acc_logger)
            print('Evaluating final model')
            evaluation(model)