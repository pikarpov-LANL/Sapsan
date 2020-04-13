import os
import click

from sapsan.core.builder.templates.algorithm import get_template as get_algorithm_template
from sapsan.core.builder.templates.dataset import get_template as get_dataset_template
from sapsan.core.builder.templates.experiment import get_template as get_experiment_template
from sapsan.core.builder.templates.runner import get_template as get_runner_template


def create_init(path: str):
    with open("{path}/__init__.py".format(path=path), "w") as file:
        file.write("")


def setup_project(name: str):
    os.mkdir(name)
    os.mkdir("./{name}/algorithm".format(name=name))
    os.mkdir("./{name}/dataset".format(name=name))
    os.mkdir("./{name}/experiment".format(name=name))

    with open("./{name}/algorithm/{name}_algorithm.py".format(name=name), "w") as file:
        file.write(get_algorithm_template(name))

    with open("./{name}/dataset/{name}_dataset.py".format(name=name), "w") as file:
        file.write(get_dataset_template(name))

    with open("./{name}/experiment/{name}_experiment.py".format(name=name), "w") as file:
        file.write(get_experiment_template(name))

    with open("./{name}/{name}_runner.py".format(name=name), "w") as file:
        file.write(get_runner_template(name))

    create_init("./{name}".format(name=name))
    create_init("./{name}/algorithm".format(name=name))
    create_init("./{name}/dataset".format(name=name))
    create_init("./{name}/experiment".format(name=name))



@click.group(help="""
    Base Sapsan cli function.
""")
def sapsan():
    pass


@sapsan.command("create")
@click.argument("name")
def create(name):
    setup_project(name=name)