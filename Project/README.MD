# Installation

### For Conda install

Install these dependencies (NumPy, OpenCV, Dlib):

```shell
conda env create --file environment.yml
#After creating environment, activate it
conda activate df_detection

# To install all packages in the existing environment from the yml
conda env update --file environment.yml --prune


conda env export --name df_detection --no-builds > environment2.yml

python -m app.main2
```
