import os
from pathlib import Path
from shared.modules.logger import info

# MUST set BEFORE importing juliacall
os.environ["JULIA_NUM_THREADS"] = "4"

JULIA_PROJECT = Path(__file__).parent.resolve()
os.environ["JULIA_PROJECT"] = str(JULIA_PROJECT)

from juliacall import Main as jl

# Activate the Julia project
jl.seval(f'include(raw"{os.path.join(JULIA_PROJECT, "src", "Fino.jl")}")')

# jl.seval('using Revise')
if not jl.seval('isdefined(Main, :Fino)'):
    jl.seval('using Fino')

info(f"Julia active project: {jl.seval('Base.active_project()')}")
# info("Julia LOAD_PATH:", jl.seval("LOAD_PATH"))
info("Fino package loaded successfully")


jl = jl

if __name__ == "__main__":
    print(jl.Fino.PricingEngine)