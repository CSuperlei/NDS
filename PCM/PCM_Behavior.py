'''
Author: CSuperlei
Date: 2025-07-15 13:54:27
LastEditTime: 2025-07-15 14:41:52
Description: 
'''
import threading
import time

import RK_Solver
import torch


def read_noise(y, device_nosie=0.1):
    return y + device_nosie

def re_programmer(device, direction="SET", old_G_0=5, v=0.03, delta_G=3):
    if direction == "SET":
        G_0 = read_noise(old_G_0 + delta_G).clamp(1, 50).float()
    elif direction == "RESET":
        G_0 = read_noise(torch.tensor(5, device=device)).clamp(1, 50).float()
    return G_0, torch.tensor(v, device=device)

class PCM:
    def __init__(self, device, interval=1e-5):
        
        self.timestr = str(time.time())
        self.device = device
        self.record_list = [] 

        self.scale_factor = torch.tensor(1e-5, dtype=torch.float64, device=device)
        self.G_0 = None
        self.v = None
        self.t_0 = None
        self.counter = 0
        self.small_counter = 0 
       
        self.updated = False
        self.timer_time = torch.tensor(0, dtype=torch.float64, device=device) 
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self.run)
        self.programming_lock = False
        self.interval = interval
        self.value = None
        self._thread.start()
        self.programming_lock = True
        self.G_0, self.v = re_programmer(self.device)
        self.last_program_time = self.timer_time.clone()
        
        self.t_0 = 7e-3
        self.G_0.to(device)
        self.v.to(device)
        self.t_0 = torch.tensor(self.t_0, dtype=torch.float64, device=device)
        self.programming_lock = False
    
    def run(self): 
        while not self._stop_event.is_set():
            time.sleep(self.interval)
            self.timer_time += self.interval
            try:
                if not self.programming_lock:
                    self.value = ((2 * self.G_0.float() - self.G_0.float() * (self.timer_time / self.t_0) ** (-self.v)) * self.scale_factor)
                    self.value = read_noise(self.value)
                    self.record_list.append((self.timer_time.item(),self.value.item(),0)) 
                    self.updated = True
            except:
                pass

    def query(self, error_ratio=None):    
        while not self.updated:
            pass

        assert self.G_0 is not None and self.v is not None and self.t_0 is not None

        self.counter += 1
        if(error_ratio is not None and error_ratio < 0.6):
            self.small_counter += 1
            if(self.small_counter > 2):
                self.re_program("SET")
        
        if(error_ratio is not None and error_ratio > 0.59):
            self.counter -= 1
        
        self.updated = False
        return self.value

    def re_program(self, direction):
        self.programming_lock = True
        self.last_program_time = self.timer_time.clone()

        self.t_0 = read_noise(torch.tensor(7e-3, device=self.device, dtype=torch.float64))

        if direction == "SET":
            self.G_0, self.v = re_programmer(self.device, direction="SET", old_G_0=self.G_0)
            self.timer_time += read_noise(torch.tensor(3e-3, device=self.device))
            self.record_list.append((self.timer_time.item(), (self.G_0.float() * self.scale_factor).item(), 1)) 

        elif direction == "RESET":
            self.G_0, self.v = re_programmer(self.device, direction="RESET", old_G_0=self.G_0)
            self.timer_time += read_noise(torch.tensor(7e-3, device=self.device))
            self.record_list.append((self.timer_time.item(), (self.G_0.float() * self.scale_factor).item(),1))
        
        assert self.G_0 > 1 and self.G_0 <= 50, "G_0 is out of range"
        
        self.counter = 0
        self.small_counter = 0
        self.programming_lock = False
    
    def adapt_stepsize(self, y, y_new, error, h_abs, step_accepted, step_rejected):
        """
        Adaptively modify the step size, use PCM.query()
        """
        self.rtol = self.rtol if self._is_iterable(self.rtol) else [self.rtol] * len(y)
        self.atol = self.atol if self._is_iterable(self.atol) else [self.atol] * len(y)

        scale = tuple(_atol + torch.max(torch.abs(_y), torch.abs(_y_new)) * _rtol for _y, _y_new, _atol, _rtol in zip(y, y_new, self.atol, self.rtol))

        error_norm = self.norm(tuple(_error / _scale for _error, _scale in zip(error, scale))).item()
       
        if error_norm < 1:
            step_accepted = True
        else:
            step_accepted = False
            step_rejected = True
            self.re_program("RESET")

        h_abs = self.query(error_ratio=error_norm)
        return h_abs, step_accepted, step_rejected
    
    def norm(self, x):
        """Compute RMS norm."""
        if torch.is_tensor(x):
            return x.norm() / (x.numel()**0.5)
        else:
            return torch.sqrt(sum(x_.norm()**2 for x_ in x) / sum(x_.numel() for x_ in x))

    def _is_iterable(self, inputs):
        try:
            iter(inputs)
            return True
        except TypeError:
            return False
