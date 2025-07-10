# Installation

### For Conda install

```shell
conda env create --file environment.yml
#After creating environment, activate it
conda activate df_detection

# To install all packages in the existing environment from the yml
conda env update --file environment.yml --prune

conda env export --name df_detection --no-builds > environment2.yml

#To run the app

python -m app.main
```
