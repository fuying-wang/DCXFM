import os
import io
import contextlib
import logging
import ipdb
import random
import pandas as pd
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

random.seed(0)


logger = logging.getLogger(__name__)

color_list = [
    [220, 20, 60],
    [119, 11, 32],
    [0, 0, 142],
    [0, 0, 230],
    [106, 0, 228],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 70],
    [0, 0, 192],
    [250, 170, 30],
    [100, 170, 30],
    [220, 220, 0],
    [175, 116, 175],
    [250, 0, 30],
    [165, 42, 42],
    [255, 77, 255],
    [0, 226, 252],
    [182, 182, 255],
    [0, 82, 0]
]


def custom_register_machx_instances(name, json_file, image_root, label_filename):
    """
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(
            json_file, timer.seconds()))

    meta = MetadataCatalog.get(name)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes

    # In COCO, certain category ids are artificially removed,
    # and by convention they are always ignored.
    # We deal with COCO's id issue and translate
    # the category ids to contiguous ids in [0, 80).

    # It works by looking at the "categories" field in the json, therefore
    # if users' own json also have incontiguous ids, we'll
    # apply this mapping as well but print a warning.
    if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
        if "coco" not in name:
            logger.warning(
                """
                Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                """
            )
    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta.thing_dataset_id_to_contiguous_id = id_map
    meta.mask_colors = color_list

    DatasetCatalog.register(name, lambda: custom_load_machx_json(
        json_file, image_root, name, label_filename))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        ignore_label=255,
        evaluator_type="machx", **meta.as_dict(),
    )


'''
FIXME:
For convinence,  we format the machx dataset into standard coco format.
However, it is important to note that this task is not instance segmentation.
'''


def custom_load_machx_json(json_file, image_root, dataset_name=None, label_filename=None,
                           extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(
            json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"]
                         for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"]
                   for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(
        len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints",
                "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    label_df = pd.read_csv(label_filename)
    label_df.rename(columns={"Unnamed: 0": "file_name"}, inplace=True)
    keys = label_df["file_name"].values
    values = label_df.loc[:, thing_classes].values
    label_dict = dict(zip(keys, values))
    # patient64648_study1_view1_frontal.png
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["lab"] = label_dict[img_dict["file_name"]]
        # in this case, we need to distinguish negative and uncertain.
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get(
                "ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(
                        poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e

            # # next: add captions for each image
            # pathology = thing_classes[obj["category_id"]]
            # observation = random.sample(prompt_dict[pathology], k=1)[0]
            # caption = f"There is {observation.lower()} indicating {pathology.lower()}."
            # obj["caption"] = caption

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


_root = os.getenv("DETECTRON2_DATASETS",
                  "/disk1/fywang/CXR_dataset/mask/MaCheX")
custom_register_machx_instances(
    name="machx_train",
    json_file=os.path.join(_root, "../backup/train_annotations.json"),
    image_root=os.path.join(_root, "images"),
    label_filename=os.path.join(_root, "../backup/train_labels.csv")
)

custom_register_machx_instances(
    name="chexlocalize_val",
    json_file=os.path.join(_root, "val_chexlocalize_annotations.json"),
    image_root=os.path.join(_root, "images"),
    label_filename=os.path.join(_root, "val_chexlocalize_labels.csv")
)