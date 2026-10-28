import inspect
import raidionicsseg.Utils.configuration_parser as cp

src = inspect.getsource(cp.ConfigResources)
print("=== ConfigResources Full Source ===")
print(src)
