import torch
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import pickle

class SAMSegmentation:
    def __init__(self,
        folder_path=None,
        image_files=None,
        lake_threshold=40,
        pixel_threshold=10,
        objects_path=None,
        output_folder=None,
        roi_index=None):
        self.pixel_threshold = pixel_threshold
        if image_files is not None:
            self.image_files = image_files
            self.folder_path = None
        else:
            self.folder_path = folder_path
            self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.lake_threshold = lake_threshold
        self.image_files.sort()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {self.device}")
        checkpoint_path_automatic = "sam2.1_hiera_base_plus.pt"
        config_path_automatic = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        checkpoint_path = "sam2.1_hiera_large.pt"
        config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_model = build_sam2(config_path_automatic, checkpoint_path_automatic, device=self.device)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam_model)
        self.predictor = SAM2ImagePredictor(build_sam2(config_path, checkpoint_path))
        self.objects_path = objects_path
        self.output_folder = output_folder
        self.roi_index = roi_index

    def lake_filter(self, masks, image):
        lago_masks = []
        for m in masks:
            seg = m["segmentation"]
            if np.count_nonzero(seg) < 100:
                continue 
            mean_color = image[seg].mean(axis=0)
            brightness = mean_color.mean()
            if brightness < self.lake_threshold:
                lago_masks.append(m)
        return lago_masks

    def get_image(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def are_masks_close(self, mask1_dict, mask2_dict):
        seg1 = mask1_dict["segmentation"]
        seg2 = mask2_dict["segmentation"]
        yx1 = np.column_stack(np.where(seg1))
        yx2 = np.column_stack(np.where(seg2))
        if yx1.size == 0 or yx2.size == 0:
            return False
        dists = cdist(yx1, yx2, metric="euclidean")
        return np.any(dists < self.pixel_threshold)

    def get_centroid(self, mask):
        cy, cx = center_of_Emass(mask.astype(np.uint8))
        return int(cx), int(cy)

    def get_lake_objects(self, idx):
        if self.objects_path is not None and os.path.exists(self.objects_path):
            with open(self.objects_path, "rb") as f:
                t = pickle.load(f)
            return t

        image = self.get_image(idx)
        masks = self.mask_generator.generate(image)
        masks = self.lake_filter(masks, image)
        masks.sort(key=lambda x: x["area"], reverse=True)
        merged_groups = []
        n = len(masks)
        visited = [False] * n

        for i in range(n):
            if visited[i]:
                continue
            group = [i]
            visited[i] = True
            queue = [i]
            while queue:
                current = queue.pop()
                for j in range(n):
                    if not visited[j] and self.are_masks_close(masks[current], masks[j]):
                        visited[j] = True
                        group.append(j)
                        queue.append(j)
            merged_groups.append(group)

        height, width = masks[0]["segmentation"].shape
        merged_mask = np.zeros((height, width), dtype=np.uint8)
        objects = {}

        for idx, group in enumerate(merged_groups, start=1):
            group_mask = np.zeros((height, width), dtype=bool)
            centroids = []
            for i in group:
                seg = masks[i]["segmentation"]
                group_mask |= seg
            merged_mask[group_mask] = idx
            centroids.append(self.get_centroid(group_mask))
            objects[idx] = {"mask": group_mask, "centroids": centroids}
        
        if self.objects_path is not None:
            with open(self.objects_path, "wb") as f:
                pickle.dump(objects, f)
        
        return objects

    def plot_objects(self, objects, imageIdx):
        image = self.get_image(imageIdx)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)  # Mostrar la imagen
        for idx, obj in objects.items():
            for cx, cy in obj["centroids"]:
                plt.scatter(cx, cy, color="red", marker="x")
                plt.text(cx + 2, cy - 2, str(idx), color="yellow", fontsize=12, weight="bold")
        plt.axis("off")
        plt.title("Segmentaciones con centroides")
        plt.savefig(f"sam_results/markers/markers_{self.roi_index}.png")
        # plt.show()
        plt.close()

    def get_coords(self, object):
        p_pos = [c for c in object["centroids"]]
        points = np.array(p_pos, dtype=np.float32)
        l_pos = [1] * len(p_pos)
        labels = np.array(l_pos, dtype=np.float32)
        return points, labels

    def segment_image(self, idx, objects):
        image = self.get_image(idx)
        self.predictor.set_image(image)
        height, width = image.shape[:2]
        final_mask = np.zeros((height, width), dtype=np.uint16)
        for idx, obj in objects.items():
            object_id = idx
            point_coords, point_labels = self.get_coords(obj)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                )
                mask = masks[0] == 1
            final_mask[mask] = object_id
        return final_mask

    def get_area_by_object(self, object_ids, mask):
        areas_dict = {}
        for object_id in object_ids:
            object_mask = mask == object_id
            area = np.count_nonzero(object_mask)
            areas_dict[object_id] = area
        return areas_dict


    def segment_images(self, objects):
        for idx, image_file in enumerate(self.image_files):
            mask = self.segment_image(idx, objects)
            
            output_path = os.path.join(self.output_folder, os.path.splitext(image_file)[0] + ".npz")
            np.savez_compressed(output_path, mask=mask)
            # print(f"Segmentación guardada {image_file}")
    
    def remove_abrupt_changes(self, df, threshold_percent=10):
        df = df.copy()
        df = df.sort_values("time")  # asegurarse de que esté ordenado

        initial_area = df['area'].iloc[0]

        # Calcular el cambio porcentual respecto al valor inicial
        df['pct_change_from_initial'] = ((df['area'] - initial_area).abs() / initial_area) * 100

        # Filtrar según el umbral
        df_clean = df[df['pct_change_from_initial'] <= threshold_percent]#.drop(columns=['pct_change_from_initial'])

        return df_clean

    def get_time_series_segmentation(self, object_ids):
        ans  = { obejct_id: [] for obejct_id in object_ids}
        
        for idx, image_file in enumerate(self.image_files):
            file_name = os.path.splitext(image_file)[0]
            mask_path = os.path.join(self.output_folder, file_name + ".npz")
            mask = np.load(mask_path)["mask"]
            areas_dict = self.get_area_by_object(object_ids, mask)
            for object_id in object_ids:
                ans[object_id].append({"area": areas_dict[object_id], "time": file_name})

        # convertir a dataframe
        for object_id in object_ids:
            df = pd.DataFrame(ans[object_id])
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            ans[object_id] = self.remove_abrupt_changes(df)

        # save ans to csv
        for object_id in object_ids:
            df = ans[object_id]
            df.to_csv(f"sam_results/areas_time_series/timeseries_{self.roi_index}_object_{object_id}.csv", index=False)	
        return ans

    def plot_time_series(self, data_dict):
        plt.figure(figsize=(7, 4.5))

        for key, df in data_dict.items():
            df = df.copy()
            df['area_hectares'] = df['area'] * 100 / 10000
            plt.plot(df['time'], df['area_hectares'], label=key, marker='o')  # <- marcador agregado

        plt.xlabel("Date")
        plt.ylabel("Area in hectare")
        # plt.title("Series temporales de área")
        # plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # put legend outside the plot
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(f"sam_results/areas_time_series/area_timeseries_{self.roi_index}.png", bbox_inches='tight')
        # plt.show()
        plt.close()
        
    
    def plot_mask(self, mask, title="Máscara segmentada"):
        # Convertir a tipo seguro
        mask = np.array(mask).astype(np.int32)

        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap="nipy_spectral")
        plt.title(title)
        plt.axis("off")
        plt.show()