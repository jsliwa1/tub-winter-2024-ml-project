import os
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import torch
from src.modules.logger import bcolors
from tqdm import tqdm


class HistoDatasetCreator:
    def __init__(
            self,
            *,
            input_path: str,
            output_path: str,
            dataset_name: str,
            slide_metadata_path: str,
            feature_path: str,
            annotations_path: str,
            label_path: str,
            color_normalization: bool,
            store_patch: bool,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.slide_metadata_path = slide_metadata_path
        self.feature_path = feature_path
        self.annotations_path = annotations_path
        self.h5_path = os.path.join(self.output_path, self.dataset_name)
        self.label_path = label_path
        self.color_normalization = color_normalization
        self.store_patch = store_patch

        self.num_normal = 0
        self.num_tumor = 0
        self.num_test = 0

        self.train_case_lst = []
        self.test_case_lst = []
        self.train_train_case_lst = []
        self.train_val_case_lst = []

        self._create_dataset()

    def _create_dataset(self):
        print(f"Creating dataset {self.dataset_name}...")
        slide_metadata = pd.read_csv(self.slide_metadata_path)

        # check whether one case has more than one slide
        ##### what to do then
        if slide_metadata["case_id"].duplicated().any():
            print("Some cases have more than one slide")
        
        # count the number of normal and tumor cases
        for _, case in enumerate(slide_metadata["case_id"]):
            if case.startswith("test"):
                self.num_test += 1
            if case.startswith("normal"):
                self.num_normal += 1
            elif case.startswith("tumor"):
                self.num_tumor += 1
        print(f"Normal cases: {self.num_normal}")
        print(f"Tumor cases: {self.num_tumor}")
        print(f"Test cases: {self.num_test}")

        data_lst = os.listdir(self.input_path)
        data_lst.sort(key=str)

        with h5py.File(self.h5_path, 'w') as h5file:
            train_group = h5file.create_group('train')
            test_group = h5file.create_group('test')

            for _, case in tqdm(enumerate(data_lst)):
                case_path = os.path.join(self.input_path, case)
                if not os.path.isdir(case_path):
                    continue
                print(f"{bcolors.OKBLUE}-------------------------------------------{bcolors.BOLD}{bcolors.ENDC}")
                print(f"{bcolors.OKBLUE}Processing {bcolors.BOLD}{case}{bcolors.ENDC}")

                if case in slide_metadata["case_id"].values:
                    print(f"Case {case} found in metadata")
                if case.startswith("test"):
                    self.test_case_lst.append(case)
                    case_group = test_group.create_group(case)
                elif case.startswith("normal") or case.startswith("tumor"):
                    self.train_case_lst.append(case)
                    case_group = train_group.create_group(case)
                else:
                    print(f"Case {case} is neither normal, tumor, nor test")
                    continue

                patch_metadata_path = os.path.join(case_path, "metadata", "df.csv")
                if os.path.exists(patch_metadata_path):
                    df = pd.read_csv(patch_metadata_path)

                    hdf5_compatible_data = self.dataframe_to_hdf5_compatible(df)
                    metadata_group = case_group.create_group("metadata")
                    for col, data in hdf5_compatible_data.items():
                        metadata_group.create_dataset(col, data=data, compression='gzip')
    
                    case_group.attrs["columns"] = list(df.columns)
                    print(f"{bcolors.OKGREEN}Metadata for {case} saved.{bcolors.ENDC}")

                patches = os.listdir(os.path.join(self.input_path, case))
                patches.sort(key=str)

                for _, patch in enumerate(patches):
                    if not patch.endswith(".jpg") or patch == "metadata":
                        continue
                    patch_path = os.path.join(case_path, patch)
                    #print(f"Processing {patch}")

                    #img = Image.open(patch_path).convert('RGB')
                    #img_array = np.array(img)
                    img_array = None

                    patch_name = os.path.splitext(patch)[0]
                    if 'path' in df.columns and f"{case}/{patch_name}.jpg" in df['path'].values:
                        patch_metadata = df.loc[df['path'] == f"{case}/{patch_name}.jpg"].iloc[0].to_dict()
                    else:
                        patch_metadata = {}

                    patch_group = case_group.create_group(patch_name)

                    # either store patches or not
                    if self.store_patch:
                        patch_group.create_dataset("patch", data=img_array, compression='gzip')

                    for key, value in patch_metadata.items():
                        patch_group.attrs[key] = value
                
                # add precomputed feature matrix
                # feature_path = os.path.join(self.feature_path, f"{case}.pt")
                # if os.path.exists(feature_path):
                #     print(f"Loading feature matrix for {case} from {feature_path}")
                #     feature_matrix = torch.load(feature_path, weights_only=True).numpy()

                #     features_group = case_group.create_group("features")
                #     features_group.create_dataset("feature_matrix", data=feature_matrix, compression='gzip')
                #     print(f"{bcolors.OKGREEN}Features for {case} saved.{bcolors.ENDC}")
                
                #add annotations slide level
                annotations_path = os.path.join(self.annotations_path, f"{case}.png")
                if os.path.exists(annotations_path):
                    print(f"Loading annotations for {case} from {annotations_path}")
                    annotations = Image.open(annotations_path)
                    #annotations_array = np.array(annotations)
                    width, height = annotations.size
                    resolution = (width, height)

                    annotations_group = case_group.create_group("annotation")
                    annotations_group.create_dataset("resolution", data=resolution, compression='gzip')
                    print(f"{bcolors.OKGREEN}Resolution for {case} saved.{bcolors.ENDC}")
                
                # add label
                label_df = pd.read_csv(self.label_path)
                if case in label_df["case_id"].values:
                    label = label_df.loc[label_df["case_id"] == case].iloc[0]["type"]
                    cls = label_df.loc[label_df["case_id"] == case].iloc[0]["class"]
                    case_group.attrs["label"] = label
                    case_group.attrs["class"] = cls
                    print(f"{bcolors.OKGREEN}Label for {case} saved.{bcolors.ENDC}")

        h5file.close()

        print(f"Train cases processed: {len(self.train_case_lst)}")
        print(f"Test cases processed: {len(self.test_case_lst)}")



    
    def dataframe_to_hdf5_compatible(self, df):
        hdf5_compatible_data = {}
        for col in df.columns:
            if df[col].dtype == 'O': 
                hdf5_compatible_data[col] = df[col].astype('S').to_numpy() 
            else:
                hdf5_compatible_data[col] = df[col].to_numpy()
        return hdf5_compatible_data

               
if __name__ == "__main__":
    input_path = "/home/space/datasets/camelyon16/patches/20x"
    output_path = "/home/pml06/dev/attdmil/HistoData/"
    dataset_name = "camelyon16_meta.h5"
    slide_metadata_path = "/home/space/datasets/camelyon16/metadata/v001/slide_metadata.csv"
    feature_path = "/home/space/datasets/camelyon16/features/20x/ctranspath_pt"
    annotations_path = "/home/space/datasets/camelyon16/annotations"
    label_path = "/home/space/datasets/camelyon16/metadata/v001/case_metadata.csv"

    hdc = HistoDatasetCreator(
        input_path=input_path,
        output_path=output_path,
        dataset_name=dataset_name,
        slide_metadata_path=slide_metadata_path,
        feature_path=feature_path,
        annotations_path=annotations_path,
        label_path=label_path,
        color_normalization=False,
        store_patch=False
    )
    print("Done.")