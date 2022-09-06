import os 
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from loss import geometry_loss_schedule

print(geometry_loss_schedule(0))
print(geometry_loss_schedule(100000))
print(geometry_loss_schedule(50000))
