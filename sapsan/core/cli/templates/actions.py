TEST_TEMPLATE = """name: Tests

on: [push]

jobs:
  build:

    runs-on: ubuntu-latest
    strategy:
      max-parallel: 4
      matrix:
        python-version: [3.6, 3.7]

    steps:
    - uses: actions/checkout@v1
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v1
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Test with pytest
      run: |
        pip install pytest
        pytest
"""

RELEASE_DRAFTER_WORKFLOW_TEMPLATE = """name: Release Drafter

on:
  push:
    branches:
      - master

jobs:
  update_release_draft:
    runs-on: ubuntu-latest
    steps:
      - uses: release-drafter/release-drafter@v5
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
"""

RELEASE_DRAFTER_TEMPLATE = """name-template: 'v$NEXT_PATCH_VERSION'
tag-template: 'v$NEXT_PATCH_VERSION'
categories:
  - title: 'Features'
    labels:
      - 'feature'
      - 'enhancement'
  - title: 'Bug Fixes'
    labels:
      - 'fix'
      - 'bugfix'
      - 'bug'
  - title: 'Maintenance'
    label: 'chore'
change-template: '- $TITLE @$AUTHOR (#$NUMBER)'
template: |
  ## Changes

  $CHANGES
"""

PYPI_TEMPLATE = """name: Publish Python distributions package to PyPI

on:
  release:
    types: [published]

jobs:
  build-n-publish:
    name: Build and publish Python distributions package to PyPI
    runs-on: ubuntu-18.04
    env:
      release_version: ${{ github.event.release.tag_name }}

    steps:
    - uses: actions/checkout@master
    - name: Set up Python 3.7
      uses: actions/setup-python@v1
      with:
        python-version: 3.7
    - name: Install wheel
      run: >-
        python -m
        pip install
        --user
        --upgrade
        setuptools wheel
    - name: Set version for release
      run: >-
        rm version &&
        echo $release_version > version &&
        echo $release_version
    - name: Build a binary wheel and a source tarball
      run: >-
        python
        setup.py
        sdist bdist_wheel
    - name: Publish distribution package to PyPI
      uses: pypa/gh-action-pypi-publish@master
      with:
        password: ${{ secrets.pypi_password }}
"""

