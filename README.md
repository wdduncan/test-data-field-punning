# test-data-field-punning
Tests the utility of OWL punning for representing relational/tabular data.

## Instructions for creating a custom Jupyter kernal to run notebooks
Content taken from [here](https://janakiev.com/blog/jupyter-virtual-envs/).

### Steps to install kernal
- create virtual environment (e.g., `python3 -m venv venv`)
- install ipykernal if not installed already:
  `python3 -m pip install --user ipykernel`
- install the the ipykernal to the virtual environment:  
  `python -m ipykernel install --user --name=<name of kernal>`
- should recieve message of the form:  
  `Installed kernelspec test-data-field-punning in <path to kernal>`
  
### Steps to delete kernal
- list kernals if needed:  
   `jupyter kernelspec list`
- delete kernal:  
  `jupyter kernelspec uninstall <name of kernal`
