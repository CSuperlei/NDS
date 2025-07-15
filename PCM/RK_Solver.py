'''
Author: CSuperlei
Date: 2025-07-15 13:11:50
LastEditTime: 2025-07-15 14:41:16
Description: 
'''
import numpy as np


class RK12():
     def step(self, func, t, dt, y, return_variables=False):
        k1 = func(t, y)
        k2 = func(t + dt, tuple(_y + 1.0 * dt * _k1 for _y, _k1 in zip(y, k1)))
        out1 = tuple(_y + _k1 * 0.5 * dt + _k2 * 0.5 * dt for _y, _k1, _k2 in zip(y, k1, k2))
        error = tuple(0.5 * dt * (_k1 - _k2) for _k1, _k2 in zip(k1, k2))

        if return_variables:
            return out1, error, [k1, k2]
        else:
            return out1, error

class RK23():
    def step(self, func, t, dt, y, return_variables=False):

        k1 = func(t, y)
        k2 = func(t + dt / 2.0, tuple(_y + 1.0 / 2.0 * dt * _k1 for _y, _k1 in zip(y, k1))  )
        k3 = func(t + dt * 0.75, tuple( _y + 0.75 * dt * _k2 for _y, _k2 in zip(y, k2))    )
        k4 = func(t + dt, tuple( _y + 2. / 9. * dt * _k1 + 1. / 3. * dt * _k2 + 4. / 9. * dt * _k3 for _y, _k1, _k2, _k3 in zip(y, k1, k2, k3))  )
        out1 = tuple( _y + 2. / 9. * dt * _k1 + 1. / 3. * dt * _k2 + 4. / 9. * dt * _k3 for _y, _k1, _k2, _k3 in zip(y, k1, k2, k3))
        error = tuple( 5/72 * dt * _k1 - 1/12 * dt * _k2 -1/9 * dt * _k3 + 1/8 * dt * _k4 for _k1, _k2, _k3, _k4 in zip(k1, k2, k3, k4))

        if return_variables:
            return out1, error, [k1, k2, k3, k4]
        else:
            return out1, error     

class RK4():
    def step(self, func, t, dt, y, return_variables=False):
        k1 = func(t, y)
        k2 = func(t + dt / 4.0, tuple(_y + 1.0 / 4.0 * dt * _k1 for _y, _k1 in zip(y, k1)))
        k3 = func(t + dt / 4.0, tuple(_y + 1.0 / 4.0 * dt * _k2 for _y, _k2 in zip(y, k2)))
        k4 = func(t + dt / 2.0, tuple(_y + 1.0 / 2.0 * dt * _k3 for _y, _ke in zip(y, k3)))
        k5 = func(t + dt * 3.0 / 4.0, tuple(_y + 3.0 / 4.0 * dt * _k4 for _y, _k4 in zip(y, k4)))
        out1 = tuple(_y + dt / 6.0 * (_k1 + 2.0 * _k2 + 2.0 * _k3 + _k4) for _y, _k1, _k2, _k3, _k4 in zip(y, k1, k2, k3, k4))
        out2 = tuple(_y + dt / 360.0 * (_k1 + 4.0 * _k2 + 6.0 * _k3 + 4.0 * k4 + _k5) for _y, _k1, _k2, _k3, _k5 in zip(y, k1, k2, k3, k5))
        error = abs(tuple(_y - _o for _y, _o in zip(out1, out2)))
        
        if return_variables:
            return out1, error, [k1, k2, k3, k4, k5]
        else:
            return out1, error