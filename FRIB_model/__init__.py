__version__ = '1.0.0'
__version_descriptions__ = {
    '1.0.0':['2024-03-05',
             'first. FRIB modeling application',
             ],
}
print(f'FRIB_model version: {__version__}. updated on {__version_descriptions__[__version__][0]}')

from . import utils
from . import flame_helper
from . import machine_portal_helper
from . import ISAAC_helper
from . import BPMQ