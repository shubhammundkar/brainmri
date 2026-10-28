import os
import sys
import raidionicsseg

def inspect_details():
    print("=== RAIDIONICSEG EXPORTS ===")
    print("dir(raidionicsseg):", [x for x in dir(raidionicsseg) if not x.startswith('__')])

    pkg_dir = os.path.dirname(raidionicsseg.__file__)
    
    # Check Utils module
    try:
        from raidionicsseg.Utils import configuration_parser
        print("\nconfiguration_parser available!")
    except Exception as e:
        print("configuration_parser error:", e)

    # Check fit module
    try:
        from raidionicsseg.fit import run_model, run_model_cli
        print("\nrun_model exported from raidionicsseg.fit!")
    except Exception as e:
        print("fit.py import error:", e)
        try:
            import raidionicsseg.fit as fit_mod
            print("dir(fit_mod):", [x for x in dir(fit_mod) if not x.startswith('__')])
        except Exception as e2:
            print("fit_mod error:", e2)

    # Search for model configuration templates or registries
    for root, dirs, files in os.walk(pkg_dir):
        for file in files:
            if file.endswith('.json') or file.endswith('.ini') or file.endswith('.py'):
                path = os.path.join(root, file)
                rel = os.path.relpath(path, pkg_dir)
                if any(k in rel.lower() for k in ['model', 'config', 'label', 'template']):
                    print(" Found file:", rel)

if __name__ == '__main__':
    inspect_details()
