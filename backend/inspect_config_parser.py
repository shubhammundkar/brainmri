import os
import inspect
import raidionicsseg.Utils.configuration_parser as cp

print("=== CONFIGURATION PARSER INSPECTION ===")
print("Classes & Functions in configuration_parser:")
for name, obj in inspect.getmembers(cp):
    if inspect.isclass(obj) or inspect.isfunction(obj):
        print(" -", name)

print("\nConfigResources init signature:", inspect.signature(cp.ConfigResources.__init__))
print("ConfigResources parse_config signature:", inspect.signature(cp.ConfigResources.parse_config) if hasattr(cp.ConfigResources, 'parse_config') else "No parse_config")

# Let's inspect source code of parse_config or __init__
try:
    src = inspect.getsource(cp.ConfigResources)
    print("\n--- ConfigResources Source Code Snippet ---")
    print("\n".join(src.split('\n')[:50]))
except Exception as e:
    print("Could not get source:", e)
