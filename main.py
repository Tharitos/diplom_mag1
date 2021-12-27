import os
#import model.py

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pixellib.instance import instance_segmentation
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def object_detection_on_an_image():
    segment_image = instance_segmentation()
    segment_image.load_model("C:/Users/Mnwa/PycharmProjects/NIR_6.1/ves.h5")

    target_class = segment_image.select_target_classes(person=True)

    segment_image.segmentImage(
        image_path="12pic.jpg",
        # image_path="2cars_people.jpeg",
        #image_path="3silicon_valley.jpg",
        show_bboxes=True,
        #segment_target_classes=target_class,
        # extract_segmented_objects=True,
        # save_extracted_objects=True,
        output_image_name="output.jpg"
    )

    # print(result[0]["scores"])
def main():
    object_detection_on_an_image()


if __name__ == '__main__':
    main()

