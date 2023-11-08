"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import os
from typing import Union, Any, Optional, List
import torch


class CheckPointer:
    """
    this code only does save/reload, user code is responsible for deciding when to do so, decide label, etc.
    user code is responsible for passing a 'label' (optional) for checkpoint

    ##########
    Note: to test model reload after preemption on slurm, it is possible to use the 'scontrol requeue' command.
    Or in the debugger, simply stop and relaunch program.
    ##########
    """

    def __init__(self, name: str, save_dir: str, verbose=True):
        self.name = name
        self.save_dir = save_dir
        self.verbose = verbose
        os.makedirs(self.save_dir, exist_ok=True)

    def save(self, module: Union[torch.nn.Module, torch.nn.DataParallel], optimizer: torch.optim.Optimizer,
             label: Optional[str] = None, current_iter: Optional[int] = None, other: Optional[Any] = None):
        """
        typically, other can be a dict of other thing to be reloaded, including for example current_best value
        of validation metric, other iteration counters, or any state to be maintained
        """

        filename_full = os.path.join(self.save_dir, self.name + '.' + label)
        try:
            # if net is a DataParallel object, save its module
            module = module.module
        except AttributeError:
            pass
        assert isinstance(module, torch.nn.modules.Module)
        chk = {'net': module.state_dict(),
               'optimizer': optimizer.state_dict() if optimizer is not None else None,
               'current_iter': current_iter,
               'other': other}
        torch.save(chk, filename_full)

    def load(self, unloaded_module: Union[torch.nn.Module, torch.nn.DataParallel],
             unloaded_optimizer: Optional[torch.optim.Optimizer] = None, label: Optional[Union[List[str], str]] = None,
             filename_full: Optional[str] = None, allow_not_exist=False, map_location: str = None):
        """
        If label is a list, try all labels in it until a corresponding filename is found.
        """
        if label is None:
            label = ''
        if not type(label) is list:
            label = [label]

        # except if filename is specified (specific checkpoint) we want to reload the default specified checkpoint
        if filename_full is None:
            filenames_full = [os.path.join(self.save_dir, self.name + '.' + l) for l in label]
        else:
            filenames_full = [filename_full]

        for f in filenames_full:
            try:
                if self.verbose:
                    print(f'trying to load: {f}')
                chk = torch.load(f, map_location=map_location)
                unloaded_module.load_state_dict(chk['net'])
                if unloaded_optimizer is not None:
                    unloaded_optimizer.load_state_dict(chk['optimizer'])
                # return stuff that the user should take care of resetting him/herself
                if self.verbose:
                    print('loaded checkpoint: {}'.format(f))
                return chk['current_iter'], chk['other']
            except FileNotFoundError:
                continue

        # if reaching here, no checkpoint was found.
        if allow_not_exist:
            if self.verbose:
                print('No checkpoint found. Skipping model reload. ')
            return None, None
        else:
            raise FileNotFoundError('No checkpoint found. ')

    def delete(self, label: Optional[str] = None, filename_full: Optional[str] = None, allow_not_exist: bool = False):
        # delete specified checkpoint
        if label is None:
            label = ''
        if filename_full is None:
            filename_full = os.path.join(self.save_dir, self.name + '.' + label)
        try:
            os.remove(filename_full)
            if self.verbose:
                print(f'Removed model: {filename_full}')
        except FileNotFoundError as e:
            if allow_not_exist:
                print(f'No such model found: {filename_full}. Skipping delete operation. ')
            else:
                raise e
