"""Setup script — overrides auto-discovery to avoid flat-layout errors."""

from setuptools import setup, find_packages

setup(
    packages=find_packages(include=["agentforge", "agentforge.*"]),
)
