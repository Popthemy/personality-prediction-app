import xtf
import inspect

output = []

def log(msg):
    output.append(str(msg))

log("=== xtf package directory ===")
log(dir(xtf))

try:
    from xtf import Router
    log("=== Router Functions ===")
    for name, member in inspect.getmembers(Router, predicate=inspect.isfunction):
        sig = inspect.signature(member)
        log(f"Function: {name}{sig}")
except Exception as e:
    log(f"Error inspecting Router: {e}")

try:
    log("=== Module level members ===")
    for name, member in inspect.getmembers(xtf, predicate=inspect.isclass):
        log(f"Class: {name}")
except Exception as e:
    log(f"Error inspecting classes: {e}")

with open("xtf_inspection.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Dump done.")
