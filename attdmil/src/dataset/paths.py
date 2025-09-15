# histopathology camelyon16
INPUT_PATH_CAMELYON16 = "/home/space/datasets/camelyon16/patches/20x"
OUTPUT_PATH_CAMELYON16 = "/home/pml06/dev/attdmil/HistoData/"
DATASET_NAME_CAMELYON16 = "camelyon16.h5"
SLIDE_METADATA_PATH_CAMELYON16 = "/home/space/datasets/camelyon16/metadata/v001/slide_metadata.csv"
FEATURE_PATH_CAMELYON16 = "/home/space/datasets/camelyon16/features/20x/ctranspath_pt"
ANNOTATIONS_PATH_CAMELYON16 = "/home/space/datasets/camelyon16/annotations"
LABEL_PATH_CAMELYON16 = "/home/space/datasets/camelyon16/metadata/v001/case_metadata.csv"

# histopathology tcga
# type can be {"blca", "brca","coad", "gbm", "hnsc", "kirc", "lgg", "luad", "lusc", "read", "stad"}
TYPE = "lgg"
INPUT_PATH_TCGA = "/home/space/datasets/tcga/%s/patches/20x" % TYPE
SLIDES_PATH_TCGA = "/home/space/datasets/tcga/%s/slides" % TYPE
ANNOTATIONS_PATH_TCGA = ""
SLIDE_IDS_TCGA = "/home/space/datasets/tcga/%s/metadata/preprocessing/pp_batch_8.json" % TYPE
SPLITS_PATH_TCGA = "/home/space/datasets/tcga/splits"
