import os
import click

from sapsan.core.cli.templates.algorithm import get_template as get_algorithm_template
from sapsan.core.cli.templates.dataset import get_template as get_dataset_template
from sapsan.core.cli.templates.experiment import get_template as get_experiment_template
from sapsan.core.cli.templates.runner import get_template as get_runner_template
from sapsan.core.cli.templates.readme import get_readme_template
from sapsan.core.cli.templates.docker import get_dockerfile_template
from sapsan.core.cli.templates.setup import get_setup_template
from sapsan.core.cli.templates.test import get_template as get_tests_template
from sapsan.core.cli.templates.actions import (TEST_TEMPLATE, PYPI_TEMPLATE,
                                               RELEASE_DRAFTER_TEMPLATE, RELEASE_DRAFTER_WORKFLOW_TEMPLATE)


def create_init(path: str):
    with open("{path}/__init__.py".format(path=path), "w") as file:
        file.write("")


def setup_project(name: str):
    os.mkdir(name)
    os.mkdir('./{name}/.github'.format(name=name))
    os.mkdir('./{name}/.github/workflows'.format(name=name))
    os.mkdir("./{name}/tests/".format(name=name))
    os.mkdir("./{name}/{name}".format(name=name))
    os.mkdir("./{name}/{name}/algorithm".format(name=name))
    os.mkdir("./{name}/{name}/dataset".format(name=name))
    os.mkdir("./{name}/{name}/experiment".format(name=name))
    click.echo("Folders has been created.")

    create_init("./{name}/tests".format(name=name))
    create_init("./{name}/{name}".format(name=name))
    create_init("./{name}/{name}/algorithm".format(name=name))
    create_init("./{name}/{name}/dataset".format(name=name))
    create_init("./{name}/{name}/experiment".format(name=name))
    click.echo("Marked folders as packages.")

    with open("./{name}/version".format(name=name), "w") as file:
        file.write("0.0.1")
        click.echo("Created version file.")

    with open("./{name}/{name}/algorithm/{name}_algorithm.py".format(name=name), "w") as file:
        file.write(get_algorithm_template(name))
        click.echo("Created algorithm file.")

    with open("./{name}/{name}/dataset/{name}_dataset.py".format(name=name), "w") as file:
        file.write(get_dataset_template(name))
        click.echo("Created dataset file.")

    with open("./{name}/{name}/experiment/{name}_experiment.py".format(name=name), "w") as file:
        file.write(get_experiment_template(name))
        click.echo("Created experiment file.")

    with open("./{name}/{name}_runner.py".format(name=name), "w") as file:
        file.write(get_runner_template(name))
        click.echo("Created runner file.")

    with open("./{name}/Dockerfile".format(name=name), "w") as file:
        file.write(get_dockerfile_template(name))
        click.echo("Created docker file.")

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
        requirements="""numpy==1.17.3
        sapsan==0.0.3
        """
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
    Base Sapsan cli function.
""")
def sapsan():
    click.echo("========================================================")
    click.echo("Lead the train to the frontiers of knowledge, my friend!")
    click.echo("========================================================")


@sapsan.command("create")
@click.argument("name")
def create(name):
    setup_project(name=name.lower())
