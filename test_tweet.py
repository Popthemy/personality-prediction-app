import inspect
from xtf import Tweet

output = []

def log(msg):
    output.append(str(msg))

log("=== Tweet Class Members ===")
for name, member in inspect.getmembers(Tweet):
    if not name.startswith('_'):
        log(f"Field: {name}")

# Also check for typing/dataclasses fields or how properties are defined
import dataclasses
if dataclasses.is_dataclass(Tweet):
    log("=== Tweet Dataclass Fields ===")
    for field in dataclasses.fields(Tweet):
        log(f"{field.name}: {field.type}")
else:
    # Just inspect annotations
    log("=== Tweet Annotations ===")
    log(getattr(Tweet, '__annotations__', 'No annotations'))

with open("xtf_tweet_schema.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Tweet inspection done.")
