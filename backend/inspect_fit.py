import os
import sys
import inspect
from raidionicsseg.fit import run_model
from raidionicsseg.Utils.configuration_parser import ConfigResources

def inspect_fit_api():
    print("=== RAIDIONICSEG FIT API INSPECTION ===")
    print("run_model signature:", inspect.signature(run_model))
    print("\nConfigResources methods/properties:")
    cr = ConfigResources.instance()
    for attr in dir(cr):
        if not attr.startswith('_'):
            print(" -", attr)

if __name__ == '__main__':
    inspect_fit_api()
