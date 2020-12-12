from sklearn import tree
import wandb
import os
import matplotlib.pyplot as plt
import subprocess
import matplotlib
import pandas as pd
import joblib

# ============================================= Functions ==============================================================


def path(path:str) -> str:
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    f_path = os.path.join(MAIN_DIR, path)
    return f_path


def local_load(path):

    data = joblib.load(path)
    print(f'LOCAL LOAD: Data successfully Loaded for {path}')
    return data


def local_save(data, dir_path, file_name: str, overwrite=False, return_path=False):

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    if not file_name.endswith('.joblib'):
        file_name = f'{file_name}.joblib'

    if not overwrite:
        if os.path.exists(os.path.join(dir_path, file_name)):
            print(f'The File {file_name} Already Exists! Creating a new version...')
            i = 2
            new_file_name = file_name
            while os.path.exists(os.path.join(dir_path, new_file_name)):
                new_file_name = f'v{i}_{file_name}'
                i += 1
            file_name = new_file_name
            print(f'New version for file: {file_name}')

    joblib.dump(data, os.path.join(dir_path, file_name))
    print(f'Local Save {file_name}: Data Successfully saved')

    if return_path:
        return Obj(
            path=os.path.join(dir_path, file_name),
            file_name=file_name,
        )


def local_data_save(data, file_name:str):
    # Use local save but with predefined save path for data
    local_save(data, file_name=file_name, dir_path=path('Data/processed'))


def local_data_load(file_name):
    # Use local load but with predefined load path for data
    file_path = os.path.join(path('Data/processed'), file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Could not find {file_name}. Local Data Searches specifically project data folder {path("Data/processed")}. Use local_load directly if you are looking for a file in a different folder')

    return local_load(file_path)


def local_model_save(model, file_name:str):
    # Use local save but with predefined save path for models
    local_save(model, file_name=file_name, dir_path=path('SavedModels'))


def local_model_load(file_name):
    # Use local load but with predefined load path for models
    file_path = os.path.join(path('SavedModels'), file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Could not find {file_name}. Local Data Searches specifically project data folder {path("Data/processed")}. Use local_load directly if you are looking for a file in a different folder')

    return local_load(file_path)

# ============================================= Classes ================================================================


class ImageSaver:

    def __init__(self, run):
        matplotlib.use('Agg')
        self.save_dir = path('../tmp')
        self.run = run

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        self.clean_up()

    def save(self, plot, name: str, format: str):

        save_path = os.path.join(self.save_dir, f'{name}.{format}')
        #TODO This needs Testing
        plot.savefig(save_path, format=format, dpi=300)
        self.run.log({name: wandb.Image(save_path)})
        self.clean_up()

    def save_graphviz(self, model: tree.DecisionTreeClassifier,
                      feature_names: list,
                      class_names: list,
                      graph_name: str,):

        name = 'tree_graph'
        format = 'dot'

        dot_out_file = os.path.join(self.save_dir, f'{name}.{format}')
        tree.export_graphviz(
            model,
            out_file=dot_out_file,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
        )
        # Convert to png
        format = 'png'
        png_out_file = os.path.join(self.save_dir, f'{name}.{format}')
        out = subprocess.run(['dot', '-Tpng', dot_out_file, '-o', png_out_file])

        self.run.log({graph_name: wandb.Image(png_out_file)})

        if out.returncode != 0:
            raise ValueError('ImageSave.save_graphviz: Graphviz dot to png command failed during subprocess run')

    def clean_up(self):
        plt.clf()
        # Clear tmp folder of files no longer needed
        files = os.listdir(self.save_dir)
        for file in files:
            os.remove(os.path.join(self.save_dir, file))


class Obj:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def compare_attributes(self, other):
        return self.__dict__ == other.__dict__

    def __call__(self,  **kwds):
        self.__dict__.update(kwds)

    def to_dict(self):
        return self.__dict__

    def to_dataframe(self, orient:str, columns=None):
        """
        Converts Anonymous object to a pandas dataframe
        :param orient: Should Object Atrribute names be the 'columns' or the 'index' of the dataframe
        :param columns: 'Only use if orient is 'index', represents the column names
        :return: Dataframe
        """
        return pd.DataFrame.from_dict(self.to_dict(), orient=orient, columns=columns)
    
