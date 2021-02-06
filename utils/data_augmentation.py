import cv2
import argparse
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique


def main():
    parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
    parser.add_argument(
        "--input_dir",
        default="",
        help="Input directory which contains images and annotations",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory which contains images and annotations after data augmentation",
        type=str,
    )
    args = parser.parse_args()

    # create the augmentor object
    PROBLEM = "instance_segmentation"
    ANNOTATION_MODE = "coco"
    INPUT_PATH = args.input_dir
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "coco"
    OUTPUT_PATH = args.output_dir + '/'
    augmentor = createAugmentor(PROBLEM, ANNOTATION_MODE, OUTPUT_MODE, GENERATION_MODE, INPUT_PATH, {"outputPath": OUTPUT_PATH})

    # add the augmentation techniques
    transformer = transformerGenerator(PROBLEM)
    for angle in [0, 90, 180, 270]:
        rotate = createTechnique("rotate", {"angle": angle})
        augmentor.addTransformer(transformer(rotate))

    flip = createTechnique("flip", {"flip": 1})
    augmentor.addTransformer(transformer(flip))

    crop = createTechnique("crop", {"percentage": 0.7, "startFrom": "TOPLEFT"})
    augmentor.addTransformer(transformer(crop))
    crop = createTechnique("crop", {"percentage": 0.7, "startFrom": "BOTTOMRIGHT"})
    augmentor.addTransformer(transformer(crop))

    # apply the augmentation process
    augmentor.applyAugmentation()
    print('Done!')


if __name__ == '__main__':
    main()
