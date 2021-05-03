import wandb
import torch
import random
import copy
from torch import nn
import pdb
import sys
sys.path.append('.')
sys.path.append('../')
from utils.distances import  L2Distance, LinfDistance
import pdb
from utils.normalize import *
normalize = CIFARNORMALIZE(32)

def evaluate_against_attacks(model, vae, attacks, val_loader, parallel=1,
                             wandb=None, num_batches=None):
    """
    Evaluates a model against the given attacks, printing the output and
    optionally writing it to a tensorboardX summary writer.
    """

    l2_distance = L2Distance()
    linf_distance = LinfDistance()

    if torch.cuda.is_available():
        l2_distance.cuda()
        linf_distance.cuda()

        device_ids = list(range(parallel))
        l2_distance = nn.DataParallel(l2_distance, device_ids)
        linf_distance = nn.DataParallel(linf_distance, device_ids)

    model_state_dict = copy.deepcopy(model.state_dict())
    for attack in attacks:
        if isinstance(attack, nn.DataParallel):
            attack_name = attack.module.__class__.__name__
        else:
            attack_name = attack.__class__.__name__

        batches_correct = []
        successful_attacks = []
        successful_l2_distance = []
        successful_linf_distance = []
        for batch_index, (inputs, labels) in enumerate(val_loader):
            if num_batches is not None and batch_index >= num_batches:
                break

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            adv_inputs = attack(inputs, labels)

            with torch.no_grad():
                logits = model(normalize(inputs) - normalize(vae(inputs)[2]))
                adv_logits = model(normalize(adv_inputs )- normalize(vae(adv_inputs)[2]))
            batches_correct.append((adv_logits.argmax(1) == labels).detach())

            success = (
                (logits.argmax(1) == labels) &  # was classified correctly
                (adv_logits.argmax(1) != labels)  # and now is not
            )

            inputs_success = inputs[success]
            adv_inputs_success = adv_inputs[success]
            num_samples = min(len(inputs_success), 1)
            adv_indices = random.sample(range(len(inputs_success)),
                                        num_samples)

            for adv_index in adv_indices:
                successful_attacks.append(torch.cat([
                    inputs_success[adv_index],
                    adv_inputs_success[adv_index],
                    torch.clamp((adv_inputs_success[adv_index] -
                                 inputs_success[adv_index]) * 3 + 0.5,
                                0, 1),
                ], dim=1).detach())

            if success.sum() > 0:
                successful_l2_distance.extend(l2_distance(
                    inputs_success,
                    adv_inputs_success,
                ).detach())
                successful_linf_distance.extend(linf_distance(
                    inputs_success,
                    adv_inputs_success,
                ).detach())
        print_cols = [f'ATTACK {attack_name}']

        correct = torch.cat(batches_correct)
        accuracy = correct.float().mean()
        if wandb is not None:
            wandb.log({f'val-{attack_name}-accuracy':accuracy.item()}, commit=False)
        print_cols.append(f'accuracy: {accuracy.item() * 100:.1f}%')

        print(*print_cols, sep='\t')

        for lpips_name, successful_lpips in [
            ('l2', successful_l2_distance),
            ('linf', successful_linf_distance)
        ]:
            if len(successful_lpips) > 0 and wandb is not None:
                wandb.log({f'val-{attack_name}-distance/{lpips_name}':
                                     wandb.Histogram(torch.stack(successful_lpips)
                                     .cpu().detach().numpy())}, commit=False)

        if len(successful_attacks) > 0 and wandb is not None:
            wandb.log({f'val-{attack_name}-images':[
                             wandb.Image(torch.cat(successful_attacks, dim=2))]}, commit=False)

        new_model_state_dict = copy.deepcopy(model.state_dict())
        for key in model_state_dict:
            old_tensor = model_state_dict[key]
            new_tensor = new_model_state_dict[key]
            max_diff = (old_tensor - new_tensor).abs().max().item()
            if max_diff > 1e-8:
                print(f'max difference for {key} = {max_diff}')
