"""
@file source.py
@author Ryan Missel

Class definition for the CosineAnnealingWarmRestarts with both Max-LR Decay and global LinearWarmup.

https://github.com/qu-gg/pytorch-cosine-annealing-with-decay-and-initial-warmup/blob/main/source.py
"""

import math
import warnings
from torch.optim.lr_scheduler import LRScheduler

class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(LRScheduler):
    """Cosine annealing with warm restarts, where each restart:
    - Has a linear warmup from eta_min to current max_lr
    - max_lr decays by 'decay' factor at each restart
    - Restart periods grow by T_mult
    """
    
    def __init__(self, optimizer, T_0, T_mult=2, eta_min=5e-8, last_epoch=-1, 
                 verbose=False, warmup_steps=350, decay=1.0):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.current_steps = 0
        self.verbose = verbose
        
        super(CosineAnnealingWarmRestartsWithDecayAndLinearWarmup, self).__init__(
            optimizer, last_epoch
        )
        
        # Store initial max learning rates
        self.initial_lrs = [group['lr'] for group in self.optimizer.param_groups]
    
    def get_lr(self):
        # Only show warning if attribute exists (newer PyTorch versions)
        if hasattr(self, '_get_lr_called_within_step') and not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        # Phase 1: Warmup from eta_min to current max_lr
        if self.current_steps < self.warmup_steps:
            warmup_factor = self.current_steps / self.warmup_steps
            return [
                self.eta_min + (base_lr - self.eta_min) * warmup_factor
                for base_lr in self.base_lrs
            ]
        
        # Phase 2: Cosine annealing from current max_lr to eta_min
        else:
            # Calculate position in cosine cycle (after warmup)
            cosine_steps = self.T_cur - self.warmup_steps
            cosine_period = self.T_i - self.warmup_steps
            
            if cosine_period <= 0:
                # Edge case: warmup longer than period
                return [self.eta_min for _ in self.base_lrs]
            
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * cosine_steps / cosine_period)) / 2
                for base_lr in self.base_lrs
            ]
    
    def step(self, epoch=None):
        """Step should be called after every batch update"""
        
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        # Increment warmup counter
        self.current_steps += 1
        
        # Check if we've completed a cycle and need to restart
        if self.T_cur >= self.T_i:
            if self.verbose:
                print(f"Restart at step {self.last_epoch}: decaying max_lr by {self.decay}")
            
            # Decay the max learning rate
            self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
            
            # Reset for new cycle
            self.T_cur = 0
            self.current_steps = 0  # Reset warmup counter
            self.T_i = self.T_i * self.T_mult  # Grow period
        
        self.last_epoch = math.floor(epoch)
        
        # Update optimizer learning rates (version-compatible way)
        if hasattr(self, '_get_lr_called_within_step'):
            self._get_lr_called_within_step = True
        
        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
            param_group['lr'] = lr
            if self.verbose:
                print(f'Epoch {epoch}: adjusting learning rate of group {i} to {lr:.4e}.')
        
        if hasattr(self, '_get_lr_called_within_step'):
            self._get_lr_called_within_step = False
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
# class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(LRScheduler):
#     r"""Set the learning rate of each parameter group using a cosine annealing
#     schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
#     is the number of epochs since the last restart and :math:`T_{i}` is the number
#     of epochs between two warm restarts in SGDR:
#     """

#     def __init__(self, optimizer, T_0, T_mult=2, eta_min=5e-8, last_epoch=-1, verbose=False, warmup_steps=350, decay=1):
#         if T_0 <= 0 or not isinstance(T_0, int):
#             raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
#         if T_mult < 1 or not isinstance(T_mult, int):
#             raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
#         self.T_0 = T_0
#         self.T_i = T_0
#         self.T_mult = T_mult
#         self.eta_min = eta_min
#         self.T_cur = last_epoch
#         super(CosineAnnealingWarmRestartsWithDecayAndLinearWarmup, self).__init__(optimizer, last_epoch, verbose)

#         # Decay attributes
#         self.decay = decay
#         self.initial_lrs = self.base_lrs

#         # Warmup attributes
#         self.warmup_steps = warmup_steps
#         self.current_steps = 0

#     def get_lr(self):
#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, "
#                           "please use `get_last_lr()`.", UserWarning)

#         return [
#             (self.current_steps / self.warmup_steps) *
#             (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
#             for base_lr in self.base_lrs
#         ]

#     def step(self, epoch=None):
#         """Step could be called after every batch update"""
#         if epoch is None and self.last_epoch < 0:
#             epoch = 0

#         if self.T_cur + 1 == self.T_i:
#             if self.verbose:
#                 print("multiplying base_lrs by {:.4f}".format(self.decay))
#             self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]

#         if epoch is None:
#             epoch = self.last_epoch + 1
#             self.T_cur = self.T_cur + 1

#             if self.current_steps < self.warmup_steps:
#                 self.current_steps += 1

#             if self.T_cur >= self.T_i:
#                 self.T_cur = self.T_cur - self.T_i
#                 self.T_i = self.T_i * self.T_mult

#         self.last_epoch = math.floor(epoch)

#         class _enable_get_lr_call:

#             def __init__(self, o):
#                 self.o = o

#             def __enter__(self):
#                 self.o._get_lr_called_within_step = True
#                 return self

#             def __exit__(self, type, value, traceback):
#                 self.o._get_lr_called_within_step = False
#                 return self

#         with _enable_get_lr_call(self):
#             for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
#                 param_group, lr = data
#                 param_group['lr'] = lr
#                 self.print_lr(self.verbose, i, lr, epoch)

#         self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class CosineAnnealingWithDecay(LRScheduler):
    """Smooth damped harmonic oscillator schedule.
    
    Behaves like a damped oscillator where:
    - Starts at max LR
    - Smoothly descends to eta_min at T_0 (first half-period)
    - Smoothly rises to decayed max at 2*T_0 (second half-period, lower peak)
    - Period increases by T_mult after each full cycle
    
    Args:
        optimizer: Wrapped optimizer
        T_0: Half-period of the first oscillation (peak to trough, in steps)
        T_mult: Factor to increase period after each full cycle (default: 1)
        eta_min: Minimum learning rate (trough of oscillation)
        decay: Factor to decay peak amplitude (e.g., 0.9 = next peak is 90%)
        last_epoch: The index of last epoch (default: -1)
        verbose: If True, prints a message to stdout for each update (default: False)
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-7, decay=0.9, 
                 last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1:
            raise ValueError(f"Expected T_mult >= 1, but got {T_mult}")
        
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay = decay
        
        super(CosineAnnealingWithDecay, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        # Current step (last_epoch starts at -1, becomes 0 after first step)
        current_step = max(0, self.last_epoch)
        
        # Find which half-period we're in
        half_period_index = 0
        steps_remaining = current_step
        current_T = self.T_0
        
        while steps_remaining >= current_T:
            steps_remaining -= current_T
            half_period_index += 1
            
            # After completing a full cycle (2 half-periods), apply T_mult
            if half_period_index % 2 == 0:
                full_cycles_completed = half_period_index // 2
                current_T = int(self.T_0 * (self.T_mult ** full_cycles_completed))
        
        full_cycle = half_period_index // 2
        is_descending = (half_period_index % 2 == 0)
        
        # Calculate amplitude
        if is_descending:
            amplitude = self.decay ** full_cycle
            phase = math.pi * steps_remaining / current_T
        else:
            amplitude = self.decay ** (full_cycle + 1)
            phase = math.pi * (1 + steps_remaining / current_T)
        
        cosine_factor = (1 + math.cos(phase)) / 2
        
        # Use base_lrs from parent class
        return [
            self.eta_min + (base_lr - self.eta_min) * amplitude * cosine_factor
            for base_lr in self.base_lrs
        ]
    
    # def step(self, epoch=None):
    #     """Step should be called after each batch update."""
    #     if epoch is None:
    #         self.steps_in_current_half += 1
    #         epoch = self.last_epoch + 1
    #     else:
    #         if epoch < 0:
    #             raise ValueError(f"Expected non-negative epoch, but got {epoch}")
    #         self._calculate_position(epoch)
        
    #     # Check if we completed current half-period
    #     if self.steps_in_current_half >= self.current_half_period_length:
    #         self.half_period_index += 1
    #         self.steps_in_current_half = 0
            
    #         # Check if we completed a FULL cycle (two half-periods)
    #         if self.half_period_index % 2 == 0:
    #             # Apply T_mult to increase period
    #             self.current_half_period_length = int(
    #                 self.T_0 * (self.T_mult ** (self.half_period_index // 2))
    #             )
        
    #     self.last_epoch = math.floor(epoch)
        
    #     # Update learning rates
    #     with _enable_get_lr_call(self):
    #         for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
    #             param_group, lr = data
    #             param_group['lr'] = lr
    #             # self.print_lr(self.verbose, i, lr, epoch)
        
    #     self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
    # def _calculate_position(self, epoch):
    #     """Calculate position for a given epoch (used when epoch is provided)."""
    #     # Reset tracking
    #     self.half_period_index = 0
    #     self.steps_in_current_half = 0
    #     self.current_half_period_length = self.T_0
        
    #     steps_remaining = epoch
        
    #     while steps_remaining >= self.current_half_period_length:
    #         steps_remaining -= self.current_half_period_length
    #         self.half_period_index += 1
            
    #         # Update half-period length after completing full cycles
    #         if self.half_period_index % 2 == 0:
    #             full_cycles_completed = self.half_period_index // 2
    #             self.current_half_period_length = int(
    #                 self.T_0 * (self.T_mult ** full_cycles_completed)
    #             )
        
    #     self.steps_in_current_half = steps_remaining


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o
    
    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self
    
    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False

#     """Smooth cosine oscillation with decaying amplitude and increasing period.
    
#     Like a damped harmonic oscillator - no jumps, just smooth continuous oscillation
#     where the peaks gradually decrease and the period gradually increases.
    
#     Args:
#         optimizer: Wrapped optimizer
#         T_0: Period of the first oscillation (in steps)
#         T_mult: Factor to increase period after each cycle (default: 1)
#         eta_min: Minimum learning rate (center of oscillation)
#         decay: Factor to decay amplitude after each cycle (e.g., 0.9 = 90% of previous)
#         last_epoch: The index of last epoch (default: -1)
#         verbose: If True, prints a message to stdout for each update (default: False)
#     """
    
#     def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-7, decay=0.9, 
#                  last_epoch=-1, verbose=False):
#         if T_0 <= 0 or not isinstance(T_0, int):
#             raise ValueError(f"Expected positive integer T_0, but got {T_0}")
#         if T_mult < 1:
#             raise ValueError(f"Expected T_mult >= 1, but got {T_mult}")
        
#         self.T_0 = T_0
#         self.T_mult = T_mult
#         self.eta_min = eta_min
#         self.decay = decay
        
#         # Track total steps for continuous oscillation
#         self.total_steps = 0
        
#         # Cycle tracking
#         self.T_cur = 0  # Steps in current cycle
#         self.T_i = T_0  # Length of current cycle
#         self.cycle = 0  # Current cycle number
#         self._initialized = False
        
#         # Store initial learning rates
        
#         # self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
#         super().__init__(optimizer, last_epoch)

#         self.base_lrs_initial = list(self.base_lrs)
#         self._initialized = True

#         # super(CosineAnnealingWithDecay, self).__init__(optimizer, last_epoch)
    
#     def get_lr(self):
#         if not getattr(self, "_initialized", False):
#             return [group["lr"] for group in self.optimizer.param_groups]

#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, "
#                          "please use `get_last_lr()`.", UserWarning)
        
#         # Calculate current amplitude (decays with each cycle)
#         amplitude = self.decay ** self.cycle
        
#         # Calculate the phase within the current cycle
#         # This creates smooth continuous oscillation
#         phase = math.pi * self.T_cur / self.T_i
        
#         # Cosine oscillation: goes from max → min → max smoothly
#         # cos(0) = 1 (start at max), cos(π) = -1 (min), cos(2π) = 1 (back to max)
#         cosine_factor = (1 + math.cos(phase)) / 2
        
#         return [
#             self.eta_min + (base_lr - self.eta_min) * amplitude * cosine_factor
#             for base_lr in self.base_lrs_initial
#         ]
    
#     def step(self, epoch=None):
#         """Step should be called after each batch update."""
#         if epoch is None:
#             epoch = self.last_epoch + 1
#             self.T_cur += 1
#         else:
#             if epoch < 0:
#                 raise ValueError(f"Expected non-negative epoch, but got {epoch}")
#             # Calculate position based on epoch
#             self._calculate_position(epoch)
            
#         self.total_steps += 1
        
#         # Check if we completed a cycle
#         if self.T_cur >= self.T_i:
#             self.cycle += 1
#             self.T_cur = 0  # Reset position in cycle
#             self.T_i = int(self.T_i * self.T_mult)  # Increase period
        
#         self.last_epoch = math.floor(epoch)
        
#         # Update learning rates
#         with _enable_get_lr_call(self):
#             for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
#                 param_group, lr = data
#                 param_group['lr'] = lr
#                 # self.print_lr(self.verbose, i, lr, epoch)
        
#         self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
#     def _calculate_position(self, epoch):
#         """Calculate which cycle and position within cycle for a given epoch."""
#         if self.T_mult == 1:
#             # Simple case: all cycles same length
#             self.cycle = epoch // self.T_0
#             self.T_i = self.T_0
#             self.T_cur = epoch % self.T_0
#         else:
#             # Complex case: cycles grow by T_mult
#             # Find which cycle we're in
#             n = 0
#             total = 0
#             T_n = self.T_0
            
#             while total + T_n <= epoch:
#                 total += T_n
#                 n += 1
#                 T_n = int(self.T_0 * (self.T_mult ** n))
            
#             self.cycle = n
#             self.T_i = T_n
#             self.T_cur = epoch - total


# class _enable_get_lr_call:
#     def __init__(self, o):
#         self.o = o
    
#     def __enter__(self):
#         self.o._get_lr_called_within_step = True
#         return self
    
#     def __exit__(self, type, value, traceback):
#         self.o._get_lr_called_within_step = False

    # def __init__(self, optimizer, T_0, eta_min=0.0000005, last_epoch=-1, gamma=0.9, T_mult=2, verbose=False):
        
    #     if T_0 <= 0 or not isinstance(T_0, int):
    #         raise ValueError("Expected positive integer T_0, but got {}".format(T_0))

    #     self.steps_per_cycle = T_0
    #     self.step_in_cycle = 0
    #     self.eta_min = eta_min
    #     self.T_cur = last_epoch
    #     self.cycle = 0
    #     self.gamma = gamma
    #     self.current_steps = 0
    #     self.T_mult = T_mult
    #     super(CosineAnnealingWithDecay, self).__init__(optimizer, last_epoch)
        
    #     self.max_lr_0 = self.base_lrs
    #     #self.max_lr = self.base_lrs

    #     # Decay attributes
    #     #self.decay = decay
    #     #self.initial_lrs = self.base_lrs

    #     # Warmup attributes
        

    # def get_lr(self):
    #     if not self._get_lr_called_within_step:
    #         warnings.warn("To get the last learning rate computed by the scheduler, "
    #                       "please use `get_last_lr()`.", UserWarning)
    #     return [
    #         (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_in_cycle / self.steps_per_cycle)) / 2)
    #         for base_lr in self.base_lrs
    #     ]

    # def step(self, epoch=None):
    #     """Step could be called after every batch update"""
    #     if epoch is None and self.last_epoch < 0:
    #         epoch = 0
        
    #     if (self.current_steps !=0 and self.current_steps % (self.steps_per_cycle*2) == 0):
    #         self.step_in_cycle = 0
    #         #self.steps_per_cycle = self.steps_per_cycle * self.T_mult

    #     if (self.current_steps !=0 and self.current_steps % self.steps_per_cycle == 0):
    #         self.cycle += 1
    #         if self.cycle % 2 != 0:
    #             self.base_lrs = self.max_lr_0
    #             self.base_lrs = [base_lr * (self.gamma ** self.cycle) for base_lr in self.base_lrs]

    #     if epoch is None:
    #         epoch = self.last_epoch + 1
        
    #     self.step_in_cycle += 1
    #     self.current_steps += 1

    #     self.last_epoch = math.floor(epoch)

    #     class _enable_get_lr_call:

    #         def __init__(self, o):
    #             self.o = o

    #         def __enter__(self):
    #             self.o._get_lr_called_within_step = True
    #             return self

    #         def __exit__(self, type, value, traceback):
    #             self.o._get_lr_called_within_step = False
    #             return self

    #     with _enable_get_lr_call(self):
    #         for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
    #             param_group, lr = data
    #             param_group['lr'] = lr
    #             self.print_lr(self.verbose, i, lr, epoch)

    #     self._last_lr = [group['lr'] for group in self.optimizer.param_groups]