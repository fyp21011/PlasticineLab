# PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics

![Docker build](https://img.shields.io/github/workflow/status/fyp21011/PlasticineLab/Docker)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/taichi)
![Taichi Dependency](https://img.shields.io/badge/taichi-<=0.7.14-orange) 


## Usage
 - Install `python3 -m pip install -e .`
 - Run `python3 -m plb.algorithms.solve --algo [algo] --env_name [env_name] --path [output-dir]`. It will run algorithms `algo` for environment `env-name` and store results in `output-dir`. For example
    `python3 -m plb.algorithms.solve --algo action --env_name Move-v1 --path output` will run call an Adam optimizer to optimize an action sequence in environment `Move-v1`

## Tune Hyper Parameters
`python3 -m plb.algorithms.tune`
For now, it would be recommended to run `tune.py` for torch_nn and lstm.


## Visualize
 - `convert -delay 20 -loop 0 output/*.png output/output.gif`
