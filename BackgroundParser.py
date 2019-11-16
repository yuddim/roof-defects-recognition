from Cvat import CvatDataset

import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
import imgaug.augmenters as iaa
from shapely.geometry import Polygon
import shapely

class SourcePreparation:

    lbl_dict = {
        "1.Vpadini": "1.Vpadini",
        "2.Vzdutie": "2.Vzdutie",
        "3.Skladki": "3.Skladki",
        "4.Zaplatki": "4.Zaplatki",
        "5.Otslaivanie_kovra": "5.Otslaivanie_kovra",
        "6.Razrivi_kovra": "6.Razrivi_kovra",
        "7.Otsutstvie_kovra": "7.Otsutstvie_kovra",
        "8.Gribok/Mox": "8.Gribok/Mox",
        "9.Rastreskivanie": "9.Rastreskivanie",
        "10.Spolzanie_kovra": "10.Spolzanie_kovra"
    }

    color_mask_dict = {
        "1.Vpadini": "00ffff",
        "2.Vzdutie": "ccff99",  # -
        "3.Skladki": "b65906",  # -
        "4.Zaplatki": "ffff00",  # not in audi #-
        "5.Otslaivanie_kovra": "ff0000",
        "6.Razrivi_kovra": "ffc1ff",  # not in audi
        "7.Otsutstvie_kovra": "8000ff",
        "8.Gribok/Mox": "5a1e1e",  # not in audi #-
        "9.Rastreskivanie": "cc99ff",
        "10.Spolzanie_kovra": "ffc125",
        "Background": "000000"
    }

    # label_list = ["traffic_sign",
    #               "car","truck",
    #               "person",
    #               "solid", "double_solid", "intermittent", "stop_lane",
    #               "traffic_light",
    #               "borders",
    #               "road",
    #               "sky"]

    #label_list = ["solid", "intermittent"]

    # label_list = ["1.Vpadini",
    #               "2.Vzdutie",
    #               "3.Skladki",
    #               "4.Zaplatki",
    #               "5.Otslaivanie_kovra",
    #               "6.Razrivi_kovra",
    #               "7.Otsutstvie_kovra",
    #               "8.Gribok/Mox",
    #               "9.Rastreskivanie",
    #               "10.Spolzanie_kovra"]

    label_list = ["1.Vpadini",
                  "2.Vzdutie",
                  "3.Skladki",
                  "4.Zaplatki",
                  "6.Razrivi_kovra",
                  "Background"]

    label_id_dict = {
        "1.Vpadini": 0,
        "2.Vzdutie": 1,
        "3.Skladki": 2,
        "4.Zaplatki": 3,
        "5.Otslaivanie_kovra": 4,
        "6.Razrivi_kovra": 5,
        "7.Otsutstvie_kovra": 6,
        "8.Gribok/Mox": 7,
        "9.Rastreskivanie": 8,
        "10.Spolzanie_kovra": 9,
        "Background": 10
    }

    def __init__(self):

        #self.path_to_cvat_instance_data = 'D:/Datasets/DefectsOnRoofs/images/Alexandr/6_poliklinika/16_6_poliklinika.xml'
        #self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Alexandr/6_poliklinika/16_6_poliklinika.xml'
        #self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Alexandr/licey/21_licey.xml'
        #self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Clynic/40_poliklinika.xml'
        #self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Silent_School/2/schola_gluhonemih.xml'
        #self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Silent_School/4/35_schola_gluhonemih.xml'
        #self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Silent_School/5/37_schola_gluhonemih.xml'
        #self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Silent_School/6/38_schola_gluhonemih.xml'
        #self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Valentin/6_poliklinika/2_6 poliklinika.xml'
        #self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Valentin/licey/3_licey.xml'
        self.path_to_cvat_semantic_data = 'D:/Datasets/DefectsOnRoofs/images/Valentin/schola_gluhonemih/11_schola_gluhonemih.xml'

        #self.background_clear_path = 'data/background/clear/'
        self.background_target_path = 'dataset/'
        #self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Alexandr/6_poliklinika/'
        #self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Alexandr/licey/'
        #self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Clynic/'
        #self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Silent_School/2/'
        #self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Silent_School/4/'
        #self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Silent_School/5/'
        #self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Andrey/Silent_School/6/'
        #self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Valentin/6_poliklinika/'
        #self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Valentin/licey/'
        self.background_source_path = 'D:/Datasets/DefectsOnRoofs/images/Valentin/schola_gluhonemih/'

        self.obj_images_path = 'dataset/images/'
        self.obj_masks_path = 'dataset/masks/'


    def convert_str_to_rgb(self, str_value):
        return (int(str_value[0:2],16), int(str_value[2:4],16), int(str_value[4:6],16))

    def rescale_polygon(self, polygon_points, sf_width, sf_height):
        polygon_array = np.array(polygon_points)
        scale_array = np.array([sf_width, sf_height])
        result_array = polygon_array * scale_array
        scaled_polygon_points = result_array.tolist()
        return scaled_polygon_points

    def generate_semantic_mask(self, sem_polygons, image_size, target_size = None):
        statistics = np.zeros((len(self.label_list)), dtype=np.int32)
        if target_size is not None:
            sf_width = target_size["width"]/image_size["width"]
            sf_height = target_size["height"] / image_size["height"]
            image_size = target_size
        else:
            sf_width = 1
            sf_height = 1
        background_color = (0, 0, 0)#self.convert_str_to_rgb(self.color_mask_dict["borders"])
        mask = np.zeros((image_size["height"],image_size["width"],3), dtype=np.uint8)
        background_points = np.array([[(0,0),(image_size["width"]-1,0),
                                            (image_size["width"]-1,image_size["height"]-1),
                                            (0, image_size["height"]-1)]], 'int32')
        mask = cv2.fillPoly(mask, background_points, background_color)
        for sem_polygon in sem_polygons:
            statistics[self.label_id_dict[sem_polygon["label"]]] = 1
            sem_label = self.lbl_dict[sem_polygon["label"]]
            sem_color = self.convert_str_to_rgb(self.color_mask_dict[sem_label])
            rescaled_polygon = self.rescale_polygon(sem_polygon["points"], sf_width, sf_height)
            roi_corners = np.asarray([rescaled_polygon], dtype=np.int32)
            mask = cv2.fillPoly(mask, roi_corners, sem_color)
        return mask, statistics

    def generate_instance_mask(self, obj_polygons, obj_boxes,
                           input_mask, image_size, target_size = None):
        if target_size is not None:
            sf_width = target_size["width"]/image_size["width"]
            sf_height = target_size["height"] / image_size["height"]
            image_size = target_size
        else:
            sf_width = 1
            sf_height = 1

        for obj_polygon in obj_polygons:
            obj_label = self.lbl_dict[obj_polygon["label"]]
            obj_color = self.convert_str_to_rgb(self.color_mask_dict[obj_label])
            rescaled_polygon = self.rescale_polygon(obj_polygon["points"], sf_width, sf_height)
            roi_corners = np.asarray([rescaled_polygon], dtype=np.int32)
            input_mask = cv2.fillPoly(input_mask, roi_corners, obj_color)
        for obj_box in obj_boxes:
            obj_label = self.lbl_dict[obj_box["label"]]
            obj_color = self.convert_str_to_rgb(self.color_mask_dict[obj_label])
            roi_corners = np.array([[(obj_box['xtl']*sf_width, obj_box['ytl']*sf_height),
                                     (obj_box['xbr']*sf_width, obj_box['ytl']*sf_height),
                                     (obj_box['xbr']*sf_width, obj_box['ybr']*sf_height),
                                     (obj_box['xtl']*sf_width, obj_box['ybr']*sf_height)]], 'int32')
            input_mask = cv2.fillPoly(input_mask, roi_corners, obj_color)
        return input_mask

    def generate_special_mask(self, sem_polygons, obj_polygons, obj_boxes,
                          label_list, image_size, is_white=True, target_size = None):
        if target_size is not None:
            sf_width = target_size["width"]/image_size["width"]
            sf_height = target_size["height"] / image_size["height"]
            image_size = target_size
        else:
            sf_width = 1
            sf_height = 1

        mask = np.zeros((image_size["height"], image_size["width"], 3), dtype=np.uint8)
        for sem_polygon in sem_polygons:
            sem_label = self.lbl_dict[sem_polygon["label"]]
            if sem_label in label_list:
                if is_white:
                    sem_color = (255, 255, 255)
                else:
                    sem_color = self.convert_str_to_rgb(self.color_mask_dict[sem_label])
                rescaled_polygon = self.rescale_polygon(sem_polygon["points"], sf_width, sf_height)
                roi_corners = np.asarray([rescaled_polygon], dtype=np.int32)
                mask = cv2.fillPoly(mask, roi_corners, sem_color)
        for obj_polygon in obj_polygons:
            obj_label = self.lbl_dict[obj_polygon["label"]]
            if obj_label in label_list:
                if is_white:
                    obj_color = (255, 255, 255)
                else:
                    obj_color = self.convert_str_to_rgb(self.color_mask_dict[obj_label])
                rescaled_polygon = self.rescale_polygon(obj_polygon["points"], sf_width, sf_height)
                roi_corners = np.asarray([rescaled_polygon], dtype=np.int32)
                mask = cv2.fillPoly(mask, roi_corners, obj_color)
        for obj_box in obj_boxes:
            obj_label = self.lbl_dict[obj_box["label"]]
            if obj_label in label_list:
                if is_white:
                    obj_color = (255, 255, 255)
                else:
                    obj_color = self.convert_str_to_rgb(self.color_mask_dict[obj_label])
                roi_corners = np.array([[(obj_box['xtl'] * sf_width, obj_box['ytl'] * sf_height),
                                         (obj_box['xbr'] * sf_width, obj_box['ytl'] * sf_height),
                                         (obj_box['xbr'] * sf_width, obj_box['ybr'] * sf_height),
                                         (obj_box['xtl'] * sf_width, obj_box['ybr'] * sf_height)]], 'int32')
                mask = cv2.fillPoly(mask, roi_corners, obj_color)
        return mask
    def generate_augmented_dataset(self, source_path, mask_path, target_path, N_per_image=15, test_percent=0.3):
        os.makedirs(target_path + '/train', exist_ok=True)
        os.makedirs(target_path + '/test', exist_ok=True)

        os.makedirs(target_path + '/train/source', exist_ok=True)
        os.makedirs(target_path + '/train/mask', exist_ok=True)
        os.makedirs(target_path + '/test/source', exist_ok=True)
        os.makedirs(target_path + '/test/mask', exist_ok=True)
        # Pipeline:
        # (1) Crop images from each side by 0-16px, do not resize the results
        #     images back to the input size. Keep them at the cropped size.
        # (2) Horizontally flip 50% of the images.
        # (3) Affine transformations
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 16)),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-5, 5),  # rotate by -5 to +5 degrees
                       translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}) # rotate by -5 to +5 degrees
        ])
        image_list = []
        mask_list = []
        file_name_list = []
        global_idx = 0
        global_list_dir = os.listdir(source_path)
        batch_size = 100
        batch_idx = 0
        for image_file_name in global_list_dir:
            global_idx += 1
            print(str(global_idx) + ' : ' + image_file_name)


            if batch_idx < batch_size:
                image_list.append(cv2.imread(os.path.join(source_path, image_file_name)))
                mask_list.append(cv2.imread(os.path.join(mask_path, image_file_name)))
                file_name_list.append(image_file_name)
                batch_idx += 1
            else:
                for count in range(N_per_image):
                    images_aug, mask_aug = seq(images=np.array(image_list,dtype=np.uint8), segmentation_maps=np.array(mask_list,dtype=np.uint8))
                    for idx in range(images_aug.shape[0]):
                        if idx > images_aug.shape[0] * (1 - test_percent):
                            folder_name = 'test'
                        else:
                            folder_name = 'train'
                        cv2.imwrite(target_path+'/'+folder_name+'/source/'+"%04d_"%(count)+file_name_list[idx],images_aug[idx])
                        cv2.imwrite(target_path+'/'+folder_name+'/mask/'+"%04d_"%(count) +file_name_list[idx], mask_aug[idx],[cv2.IMWRITE_PNG_COMPRESSION, 0])
                        print(str(count)+':' + str(idx))
                image_list = []
                mask_list = []
                file_name_list = []
                batch_idx = 0

    def all_resources_preparation(self):
        #path_to_cvat_data = self.path_to_cvat_instance_data
        #target_path = self.background_target_path

        # cvat_data = CvatDataset()
        # cvat_data.load(self.path_to_cvat_instance_data)

        cvat_data_semantic = CvatDataset()
        cvat_data_semantic.load(self.path_to_cvat_semantic_data)

        target_size = {'width': 1024, 'height': 576}

        os.makedirs(self.background_target_path + 'resized_source', exist_ok=True)
        os.makedirs(self.background_target_path+'resized_semantic',exist_ok=True)
        #os.makedirs(self.background_target_path + 'resized_instance_semantic', exist_ok=True)
        #os.makedirs(self.background_target_path + 'resized_special_markup', exist_ok=True)


        image_ids = cvat_data_semantic.get_image_ids()
        image_statistics = np.zeros((len(self.label_list)),dtype=np.int32)
        polygons = []
        polygons_semantic = []
        boxes = []
        points = []
        image_sizes = []
        image_names = []
        for idx, image in enumerate(image_ids):
            #polygons.append(cvat_data.get_polygons(image))
            polygons_semantic.append(cvat_data_semantic.get_polygons(image))
            #boxes.append(cvat_data.get_boxes(image))
            #points.append(cvat_data.get_points(image))
            image_sizes.append(cvat_data_semantic.get_size(image))
            image_names.append(cvat_data_semantic.get_name(image))

            # points_list = points[idx]
            image_size = image_sizes[idx]
            image_name = image_names[idx]
            print(image_name)



            #
            # # background creation
            # self.background_alignment(points_list=points_list, image_size=image_size, image_name=image_name)

            # cars cropping
            #obj_polygons = polygons[idx]
            sem_polygons = polygons_semantic[idx]
            #obj_boxes = boxes[idx]

            if target_size is not None:
                source_image = cv2.imread(self.background_source_path+ os.path.basename(image_name))
                sf_width = target_size["width"] / image_size["width"]
                sf_height = target_size["height"] / image_size["height"]
                newX, newY = source_image.shape[1] * sf_width, source_image.shape[0] * sf_height
                source_image = cv2.resize(source_image, (int(newX), int(newY)))
                full_target_path = self.background_target_path + '/resized_source/' + os.path.basename(image_name)
                cv2.imwrite(full_target_path,source_image)

            semantic_mask, statistics = self.generate_semantic_mask(sem_polygons=sem_polygons, image_size=image_size,
                                                        target_size=target_size)
            full_target_path = self.background_target_path+'resized_semantic/'+os.path.basename(image_name)
            cv2.imwrite(full_target_path, semantic_mask,[cv2.IMWRITE_PNG_COMPRESSION, 0])
            image_statistics = image_statistics + statistics
            #
            # instance_semantic_mask = self.generate_instance_mask(obj_polygons=obj_polygons, obj_boxes=obj_boxes,
            #                                                      input_mask=semantic_mask, image_size=image_size,
            #                                                      target_size=target_size)
            # full_target_path = self.background_target_path + 'resized_instance_semantic/' + os.path.basename(image_name)
            # cv2.imwrite(full_target_path, instance_semantic_mask,[cv2.IMWRITE_PNG_COMPRESSION, 0])


            # special_mask = self.generate_special_mask(sem_polygons=sem_polygons, obj_polygons=obj_polygons,
            #                                           obj_boxes=obj_boxes, label_list=self.label_list,
            #                                           image_size=image_size, is_white=False,
            #                                           target_size=target_size)
            # full_target_path = self.background_target_path + 'resized_special_markup/' + os.path.basename(image_name)
            # cv2.imwrite(full_target_path, special_mask,[cv2.IMWRITE_PNG_COMPRESSION, 0])

            #self.cars_cropping(car_polygons=car_polygons, image_name=image_name)

        #print(image_ids)
        print (image_statistics)

# source_preparator = SourcePreparation()

#source_preparator.all_resources_preparation()
# source_preparator.generate_augmented_dataset(source_path = "D:/Projects/Defects_Recognition/UNet_project/dataset/resized_source",
#                                              mask_path = "D:/Projects/Defects_Recognition/UNet_project/dataset/resized_semantic",
#                                              target_path = "D:/Projects/Defects_Recognition/UNet_project/dataset/augmented_dataset",
#                                              N_per_image=4, test_percent=0.2)
# source_preparator.generate_augmented_dataset(source_path = "D:/Projects/nkbvs_segmentation/dataset/resized_source",
#                                              mask_path = "D:/Projects/nkbvs_segmentation/dataset/resized_instance_semantic",
#                                              target_path = "D:/Projects/nkbvs_segmentation/dataset/augmented_multiclass_dataset")
# source_preparator.generate_augmented_dataset(source_path = "D:/Projects/nkbvs_segmentation/dataset/resized_source",
#                                              mask_path = "D:/Projects/nkbvs_segmentation/dataset/resized_special_markup",
#                                              target_path = "D:/Projects/nkbvs_segmentation/dataset/augmented_markup_dataset")


#source_preparator.all_resources_preparation()


# source_preparator.maskLabelsToTxt(mask_path='D:/Projects/RAAISummerSchool/data/dataset/test/mask',
#                                   txt_path='D:/Projects/RAAISummerSchool/data/dataset/test/labels')





