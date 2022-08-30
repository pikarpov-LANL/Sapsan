import os
import click
import jupytext
import nbformat
import pytest
import shutil

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
    with open("{path}/__init__.py".format(path=path), "w") as file:
        file.write("")

def setup_project(name: str, ddp: bool):
    click.echo("Created...")
    os.mkdir(name)
    click.echo("Project Folder:             {name}/".format(name=name))

    os.mkdir("./{name}/data".format(name=name))
    click.echo("Data Folder:                {name}/data/".format(name=name))

    with open("./{name}/{name}_estimator.py".format(name=name), "w") as file:
        file.write(get_estimator_template(name))
        click.echo("Estimator Template:         {name}/{name}_estimator.py".format(name=name))
        
    with open("./{name}/{name}.py".format(name=name), "w") as file:
        file.write(get_notebook_template(name))

    ntbk = jupytext.read("./{name}/{name}.py".format(name=name)) 
    nbformat.write(ntbk, "./{name}/{name}.ipynb".format(name=name))
    os.remove("./{name}/{name}.py".format(name=name))
    click.echo("Jupyter Notebook Template:  {name}/{name}.ipynb".format(name=name))
    
    with open("./{name}/Dockerfile".format(name=name), "w") as file:
        file.write(get_dockerfile_template(name))
        click.echo("Docker Template:            {name}/Dockerfile".format(name=name))
        
    with open("./{name}/Makefile".format(name=name), "w") as file:
        file.write(get_makefile_template(name))
        click.echo("Docker Makefile:            {name}/Makefile".format(name=name))
        
    if ddp:
        click.echo("-----")
        click.echo("Get Pytorch Backend = True: {name}/torch_backend.py".format(name=name))
        shutil.copy("{path}/lib/estimator/torch_backend.py".format(path=__path__[0]), "./{name}/".format(name=name))
    

def setup_package(name: str):
    os.mkdir(name)
    os.mkdir('./{name}/.github'.format(name=name))
    os.mkdir('./{name}/.github/workflows'.format(name=name))
    os.mkdir("./{name}/tests/".format(name=name))
    os.mkdir("./{name}/{name}".format(name=name))
    os.mkdir("./{name}/{name}/estimator".format(name=name))
    os.mkdir("./{name}/{name}/dataset".format(name=name))
    os.mkdir("./{name}/{name}/experiment".format(name=name))
    click.echo("Folders has been created.")

    create_init("./{name}/tests".format(name=name))
    create_init("./{name}/{name}".format(name=name))
    create_init("./{name}/{name}/estimator".format(name=name))
    create_init("./{name}/{name}/dataset".format(name=name))
    create_init("./{name}/{name}/experiment".format(name=name))
    click.echo("Marked folders as packages.")

    with open("./{name}/{name}/_version.py".format(name=name), "w") as file:
        file.write("__version__ = v0.0.1")
        click.echo("Created version file.")

    with open("./{name}/{name}_estimator.py".format(name=name), "w") as file:
        file.write(get_estimator_template(name))
        click.echo("Created estimator file.")
        
    with open("./{name}/{name}.py".format(name=name), "w") as file:
        file.write(get_notebook_template(name))

    ntbk = jupytext.read("./{name}/{name}.py".format(name=name)) 
    nbformat.write(ntbk, "./{name}/{name}.ipynb".format(name=name))
    os.remove("./{name}/{name}.py".format(name=name))
    click.echo("Jupyter Notebook Template to run everything from")

    with open("./{name}/{name}/dataset/{name}_dataset.py".format(name=name), "w") as file:
        file.write(get_dataset_template(name))
        click.echo("Created dataset file.")

    with open("./{name}/{name}/experiment/{name}_experiment.py".format(name=name), "w") as file:
        file.write(get_experiment_template(name))
        click.echo("Created experiment file.")

    #with open("./{name}/{name}_runner.py".format(name=name), "w") as file:
    #    file.write(get_runner_template(name))
    #    click.echo("Created runner file.")

    with open("./{name}/Dockerfile".format(name=name), "w") as file:
        file.write(get_dockerfile_template(name))
        click.echo("Created docker file.")
        
    with open("./{name}/Makefile".format(name=name), "w") as file:
        file.write(get_makefile_template(name))
        click.echo("Docker Makefile:            {name}/Makefile".format(name=name))

    with open("./{name}/setup.py".format(name=name), "w") as file:
        file.write(get_setup_template(name))
        click.echo("Created setup file.")

    with open("./{name}/README.md".format(name=name), "w") as file:
        file.write(get_readme_template(name))
        click.echo("Created readme.")

    with open("./{name}/tests/test_estimator.py".format(name=name), "w") as file:
        file.write(get_tests_template(name))
        click.echo("Created tests.")

    with open("./{name}/requirements.txt".format(name=name), "w") as file:
        requirements="sapsan = %s"%(__version__.strip("v"))
        file.write(requirements)
        click.echo("Created requirements file.")

    with open("./{name}/.github/release-drafter.yml".format(name=name), "w") as file:
        file.write(RELEASE_DRAFTER_TEMPLATE)
    with open("./{name}/.github/workflows/release-drafter.yml".format(name=name), "w") as file:
        file.write(RELEASE_DRAFTER_WORKFLOW_TEMPLATE)
    with open("./{name}/.github/workflows/pythonpackage.yml".format(name=name), "w") as file:
        file.write(TEST_TEMPLATE)
    with open("./{name}/.github/workflows/pypi-release.yml".format(name=name), "w") as file:
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
    if os.path.isdir("./{dir_name}".format(dir_name=dir_name)):
        click.echo("./sapsan_examples folder exists - please delete or try a different path")
    else:
        notebooks = ['cnn_example.ipynb', 'picae_example.ipynb','krr_example.ipynb']
        os.mkdir("./{dir_name}".format(dir_name=dir_name))
        for nt in notebooks:
            shutil.copy("{path}/{dir_name}/{nt}".format(path=__path__[0], dir_name="examples", nt=nt), 
                        "./{dir_name}/{nt}".format(dir_name=dir_name, nt=nt))
        shutil.copytree("{path}/{dir_name}/data".format(path=__path__[0],dir_name="examples"), 
                        "./{dir_name}/data".format(dir_name=dir_name))
        shutil.copytree("{path}/{dir_name}/GUI".format(path=__path__[0],dir_name="examples"), 
                        "./{dir_name}/GUI".format(dir_name=dir_name))
        click.echo("Done, check out ./sapsan_examples")   
        
@sapsan.command("get_torch_backend", help="Copy torch_backend.py to your working directory")
def get_torch_backend():    
    shutil.copy("{path}/lib/estimator/torch_backend.py".format(path=__path__[0]), "./")

@sapsan.command("gtb", help="Copy torch_backend.py to your working directory")
def gtb():    
    shutil.copy("{path}/lib/estimator/torch_backend.py".format(path=__path__[0]), "./")    