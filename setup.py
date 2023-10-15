# https://scikit-build.readthedocs.io/en/latest/usage.html

import os
import re
from skbuild import setup
from setuptools import find_packages

# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
META = {}
with open(os.path.join("src", "python", "veritas", "__init__.py")) as f:
    locs = {}
    for l in f.readlines():
        m = re.fullmatch('^__(?P<key>\\w+)__\\s*=\\s*(?P<value>.+)\\n$', l)
        if m:
            k, v = m.group("key"), m.group("value")
            exec(f"__{k}__={v}", globals(), locs)
            META[k] = locs[f"__{k}__"]
    del locs

if __name__ == "__main__":
    setup(
        name="dtai-veritas",
        license=META["license"],
        python_requires='>=3.8',
        url=META["url"],
        version=META["version"],
        author=META["author"],
        author_email=META["email"],
        maintainer=META["author"],
        maintainer_email=META["email"],
        long_description=META["doc"],
        packages=find_packages('src/python'),
        package_dir={ "": "src/python" },
        install_requires=[
            "numpy",
        ],
        zip_safe=False,
    )
