import os
import click
import jupytext
import nbformat
import pytest
import shutil
import glob

import sys
sys.path.append("../../../")

from sapsan._version import __version__
from sapsan import __path__

from sapsan.core.cli.templates.notebook import get_template as get_notebook_template
from sapsan.core.cli.templates.estimator import get_template as get_estimator_template
from sapsan.core.cli.templates.docker import get_template as get_dockerfile_template
from sapsan.core.cli.templates.makefile import get_template as get_makefile_template
from sapsan.core.cli.templates.dataset import get_template as get_dataset_template
from sapsan.core.cli.templates.experiment import get_template as get_experiment_template
from sapsan.core.cli.templates.runner import get_template as get_runner_template
from sapsan.core.cli.templates.readme import get_readme_template
from sapsan.core.cli.templates.setup import get_setup_template
from sapsan.core.cli.templates.test import get_template as get_tests_template
from sapsan.core.cli.templates.actions import (TEST_TEMPLATE, PYPI_TEMPLATE,
                                               RELEASE_DRAFTER_TEMPLATE, RELEASE_DRAFTER_WORKFLOW_TEMPLATE)


def create_init(path: str):
    with open(f"{path}/__init__.py", "w") as file:
        file.write("")

def setup_project(name: str, ddp: bool):
    click.echo("Created...")
    os.mkdir(name)
    click.echo(f"Project Folder:             {name}/")

    os.mkdir(f"./{name}/data")
    click.echo(f"Data Folder:                {name}/data/")

    with open(f"./{name}/{name}_estimator.py", "w") as file:
        file.write(get_estimator_template(name))
        click.echo(f"Estimator Template:         {name}/{name}_estimator.py")
        
    with open(f"./{name}/{name}.py", "w") as file:
        file.write(get_notebook_template(name))

    ntbk = jupytext.read(f"./{name}/{name}.py") 
    nbformat.write(ntbk, f"./{name}/{name}.ipynb")
    os.remove(f"./{name}/{name}.py")
    click.echo(f"Jupyter Notebook Template:  {name}/{name}.ipynb")
    
    with open(f"./{name}/Dockerfile", "w") as file:
        file.write(get_dockerfile_template(name))
        click.echo(f"Docker Template:            {name}/Dockerfile")
        
    with open(f"./{name}/Makefile", "w") as file:
        file.write(get_makefile_template(name))
        click.echo(f"Docker Makefile:            {name}/Makefile")
        
    if ddp:
        click.echo("-----")
        click.echo(f"Get Pytorch Backend = True: {name}/torch_backend.py")
        shutil.copy(f"{__path__[0]}/lib/estimator/torch_backend.py", f"./{name}/")
    

def setup_package(name: str):
    os.mkdir(name)
    os.mkdir(f"./{name}/.github")
    os.mkdir(f"./{name}/.github/workflows")
    os.mkdir(f"./{name}/tests/")
    os.mkdir(f"./{name}/{name}")
    os.mkdir(f"./{name}/{name}/estimator")
    os.mkdir(f"./{name}/{name}/dataset")
    os.mkdir(f"./{name}/{name}/experiment")
    click.echo("Folders has been created.")

    create_init(f"./{name}/tests")
    create_init(f"./{name}/{name}")
    create_init(f"./{name}/{name}/estimator")
    create_init(f"./{name}/{name}/dataset")
    create_init(f"./{name}/{name}/experiment")
    click.echo("Marked folders as packages.")

    with open(f"./{name}/{name}/_version.py", "w") as file:
        file.write("__version__ = v0.0.1")
        click.echo("Created version file.")

    with open(f"./{name}/{name}_estimator.py", "w") as file:
        file.write(get_estimator_template(name))
        click.echo("Created estimator file.")
        
    with open(f"./{name}/{name}.py", "w") as file:
        file.write(get_notebook_template(name))

    ntbk = jupytext.read(f"./{name}/{name}.py") 
    nbformat.write(ntbk, f"./{name}/{name}.ipynb")
    os.remove(f"./{name}/{name}.py")
    click.echo("Jupyter Notebook Template to run everything from")

    with open(f"./{name}/{name}/dataset/{name}_dataset.py", "w") as file:
        file.write(get_dataset_template(name))
        click.echo("Created dataset file.")

    with open(f"./{name}/{name}/experiment/{name}_experiment.py", "w") as file:
        file.write(get_experiment_template(name))
        click.echo("Created experiment file.")

    #with open("./{name}/{name}_runner.py".format(name=name), "w") as file:
    #    file.write(get_runner_template(name))
    #    click.echo("Created runner file.")

    with open(f"./{name}/Dockerfile", "w") as file:
        file.write(get_dockerfile_template(name))
        click.echo("Created docker file.")
        
    with open(f"./{name}/Makefile", "w") as file:
        file.write(get_makefile_template(name))
        click.echo(f"Docker Makefile:            {name}/Makefile")

    with open(f"./{name}/setup.py", "w") as file:
        file.write(get_setup_template(name))
        click.echo("Created setup file.")

    with open(f"./{name}/README.md", "w") as file:
        file.write(get_readme_template(name))
        click.echo("Created readme.")

    with open(f"./{name}/tests/test_estimator.py", "w") as file:
        file.write(get_tests_template(name))
        click.echo("Created tests.")

    with open(f"./{name}/requirements.txt", "w") as file:
        requirements="sapsan = %s"%(__version__.strip("v"))
        file.write(requirements)
        click.echo("Created requirements file.")

    with open(f"./{name}/.github/release-drafter.yml", "w") as file:
        file.write(RELEASE_DRAFTER_TEMPLATE)
    with open(f"./{name}/.github/workflows/release-drafter.yml", "w") as file:
        file.write(RELEASE_DRAFTER_WORKFLOW_TEMPLATE)
    with open(f"./{name}/.github/workflows/pythonpackage.yml", "w") as file:
        file.write(TEST_TEMPLATE)
    with open(f"./{name}/.github/workflows/pypi-release.yml", "w") as file:
        file.write(PYPI_TEMPLATE)

    click.echo("Created github actions.")


@click.group(help="""
    Base Sapsan cli function. Further descriptions and tutorials for Sapsan 
    and the CLI can be found on https://sapsan-wiki.github.io/overview/getting_started/#command-line-interface-cli-jupyter-notebooks
""")
@click.version_option(version = __version__, prog_name="Sapsan", 
                      message = "%(prog)s %(version)s")

def sapsan():
    pass
    
@sapsan.command("create", help="Sets up a new project with an estimator template")
@click.option('--name', '-n', default="new_project", show_default=True, help="name of the new project")
@click.option('--gtb/--no-gtb','--get_torch_backend', default=False, show_default=True, 
              help="Copies torch_backend.py into working directory to customize the Catalyst Runner - adjust its Distributed Data Parallel (DDP) settings")
def create(name, gtb):
    click.echo("========================================================")  
    click.echo("Lead the train to the frontiers of knowledge, my friend!")
    click.echo("========================================================")  
    setup_project(name=name.lower(), ddp=gtb)    

#@sapsan.command("create_package", help="Sets up a new package ready for pypi distribution.")
#@click.option('--name', '-n', default="new_package", show_default=True, help="name of the new package")
#def create_package(name):
#    click.echo("========================================================")  
#    click.echo("Lead the train to the frontiers of knowledge, my friend!")
#    click.echo("========================================================")  
#    setup_package(name=name.lower())
    
@sapsan.command("test", help="Run tests to check if everything is working correctly")
def test():
    pytest.main(__path__)
    
@sapsan.command("get_examples", help="Copy examples to your working directory")    
def get_examples():
    dir_name = "sapsan_examples"
    if os.path.isdir(f"./{dir_name}"):
        click.echo("./sapsan_examples folder exists - please delete or try a different path")
    else:
        os.mkdir(f"./{dir_name}")
        
        for notebook in glob.glob(f"{__path__[0]}/examples/*.ipynb"):
            shutil.copy(notebook, f"./{dir_name}/{notebook.split('/')[-1]}")
            
        shutil.copytree(f"{__path__[0]}/examples/data", f"./{dir_name}/data")
        shutil.copytree(f"{__path__[0]}/examples/GUI", f"./{dir_name}/GUI")
        
        click.echo("Done, check out ./sapsan_examples")   
        
@sapsan.command("get_torch_backend", help="Copy torch_backend.py to your working directory")
def get_torch_backend():    
    shutil.copy(f"{__path__[0]}/lib/estimator/torch_backend.py", "./")
    click.echo("Copied torch_backend.py!")

@sapsan.command("gtb", help="Copy torch_backend.py to your working directory")
def gtb():    
    shutil.copy(f"{__path__[0]}/lib/estimator/torch_backend.py", "./")  
    click.echo("Copied torch_backend.py!")  
