# save as regenerate_requirements.py
import pkg_resources

with open("requirements.txt", "w") as f:
    for dist in pkg_resources.working_set:
        f.write(f"{dist.key}=={dist.version}\n")
