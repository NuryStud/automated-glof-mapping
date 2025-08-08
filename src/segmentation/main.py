from SAMSegmentation import SAMSegmentation
import os
import traceback


# for roi_index in [29,39,60,71,94,110,111,127,136]:
id_img_to_get_markers = 0
# objects = {}
# 24
# objects[1] = {"centroids": [(326, 261)]}
# objects[2] = {"centroids": [(160, 115)]}
# objects[3] = {"centroids": [(10, 226)]}
# objects[4] = {"centroids": [(165, 591)]}

# CASO ESPECIAL LAGOS NAO EM 2024
# 31
# objects[1] = {"centroids": [(315, 171)]}
# objects[2] = {"centroids": [(133, 81)]}
# objects[3] = {"centroids": [(79, 447)]}
# objects[4] = {"centroids": [(279, 581)]}

# 34
# objects[1] = {"centroids": [(288, 82)]}

# 48
# objects[1] = {"centroids": [(179, 122)]}
# objects[2] = {"centroids": [(168, 430)]}
# objects[3] = {"centroids": [(22, 528)]}


# flag = True
# while flag or id_img_to_get_markers < 50:
for roi_index in [5]: # 24, 31, 34, 48
    try:
        output_folder = f"sam_results/images_rgb_water/{roi_index}/"
        os.makedirs(output_folder, exist_ok=True)
        sam_segmentation = SAMSegmentation(
            folder_path=f"images_rgb_water/{roi_index}/",
            objects_path=f"{output_folder}objects.pkl",
            output_folder=output_folder,
            roi_index=roi_index
        )
        objects = sam_segmentation.get_lake_objects(id_img_to_get_markers) # first image of roi_index
        # sam_segmentation.plot_objects(objects, id_img_to_get_markers)
        # sam_segmentation.segment_images(objects)
        series_dict = sam_segmentation.get_time_series_segmentation(objects)
        sam_segmentation.plot_time_series(series_dict)
        flag = False
        print(f"Success with id image: {id_img_to_get_markers}")
    except:
        id_img_to_get_markers += 1
        print(f"Error in roi_index {roi_index}")
        traceback.print_exc()