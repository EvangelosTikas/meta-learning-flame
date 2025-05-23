import setuptools
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "meta-learning-flame"
AUTHOR_USER_NAME = "EvangelosTikas"
SRC_REPO = "meta-flame"
AUTHOR_EMAIL = "vagtikas@gmail.com"


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author="Evangelos Tikas",
    author_email="vagtikas@gmail.com",
    description="A template directory for Meta-Learning Applications.",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "MLTMPLTE": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/",
    },
    packages=setuptools.find_packages(),
    install_requires=get_requirements('requirements.txt')

)
