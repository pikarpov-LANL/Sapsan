TEMPLATE = """
{name}
====

README
"""


def get_readme_template(name: str) -> str:
    return TEMPLATE.format(name=name)
