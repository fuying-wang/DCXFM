import json
import random

predefined_prompt_dict = {
    "Air Trapping": [
        "Hyperinflation of the Lungs",
        "Bilateral Radiolucency",
        "Decreased Vascular Markings",
        "Bulging of the Fissures",
        "Mosaic Attenuation Pattern"
    ],
    "Aortic Atheromatosis": [
        "Aortic Calcification",
        "Enlargement of the Aortic Silhouette",
        "Tortuosity of the Aorta",
        "Displacement of Trachea or Esophagus",
        "Irregular Aortic Contour"
    ],
    "Aortic enlargement": [
        "Widened mediastinum",
        "Enlarged aortic contour",
        "Displacement of trachea or esophagus",
        "Calcification of the aortic wall",
    ],
    "Aortic Elongation": [
        "Widened Mediastinum",
        "Prominent Aortic Knob",
        "Tortuous Aortic Contour",
        "Displacement of Trachea or Esophagus",
        "Altered Cardiac Silhouette"
    ],
    "Atelectasis": [
        "Increased opacity",
        "Volume loss of the affected lung region",
        "Displacement of the diaphragm",
        "Blunting of the costophrenic angle",
        "Shifting of the mediastinum"
    ],
    "Bronchiectasis": [
        "Dilated and Thickened Bronchial Walls",
        "Increased Lung Markings",
        "Honeycombing Pattern",
        "Areas of Increased Opacity",
        "Volume Loss in Affected Areas"
    ],
    "Calcification": [
        "Calcified vascular",
        "Small, rounded opacities",
        "Calcified lymph nodes",
        "Calcified pleural plaques",
        "Calcified lung masses"
    ],
    "Cardiomegaly": [
        "Increased size of the heart shadow",
        "Enlargement of the heart silhouette",
        "Increased diameter of the heart border",
        "Increased cardiothoracic ratio"
    ],
    "Consolidation": [
        "Loss of lung volume",
        "Increased density of lung tissue",
        "Obliteration of the diaphragmatic silhouette",
        "Presence of opacities"
    ],
    "Costophrenic Angle Blunting": [
        "Loss of Sharpness of Costophrenic Angles",
        "Pleural Effusion Indicators",
        "Meniscus Sign",
        "Volume Discrepancy Between Lung Fields",
        "Elevation of the Hemidiaphragm"
    ],
    "Edema": [
        "Blurry vascular markings in the lungs",
        "Enlarged heart",
        "Kerley B lines",
        "Increased interstitial markings in the lungs",
        "Widening of interstitial spaces",
    ],
    "Emphysema": [
        "Flattened hemidiaphragm",
        "Pulmonary bullae",
        "Hyperlucent lungs",
        "Horizontalisation of ribs",
        "Barrel Chest",
    ],
    "Enlarged Cardiomediastinum": [
        "Increased width of the heart shadow",
        "Widened mediastinum",
        "Abnormal contour of the heart border",
        "Fluid or air within the pericardium",
        "Mass within the mediastinum",
    ],
    "Flattened Diaphragm": [
        "Decreased Diaphragmatic Curvature",
        "Low Diaphragmatic Position",
        "Increased Retrosternal Air Space",
        "Barrel Chest Appearance",
        "Elevation of the Hemidiaphragm"
    ],
    "Fracture": [
        "Discontinuity or Irregularity in Bone Structure",
        "Displacement of Bone Fragments",
        "Angulation or Deformity",
        "Increased Density or 'Fuzzy' Appearance at Fracture Site",
        "Associated Soft Tissue Swelling"
    ],
    "Granuloma": [
        "Well-defined, Dense Nodule(s)",
        "Small nodules",
        "Nodules in the upper lobes",
        "Stable nodules",
        "Lack of Associated Lung Changes"
    ],
    "Hemidiaphragm Elevation": [
        "Asymmetry of the Diaphragmatic Domes",
        "Reduced Lung Volume on the Elevated Side",
        "Shift of Mediastinal Structures",
        "Displacement of Abdominal Contents",
        "Compensatory Hyperexpansion of the Opposite Lung"
    ],
    "Hernia": [
        "Bulge or swelling in the abdominal wall",
        "Protrusion of intestine or other abdominal tissue",
        "Swelling or enlargement of the herniated sac or surrounding tissues",
        "Retro-cardiac air-fluid level",
        "Thickening of intestinal folds"
    ],
    "Hilar Enlargement": [
        "Increased Hilar Shadow",
        "Asymmetry of the Hilar Regions",
        "Prominent Vascular Markings",
        "Mass Effect on Adjacent Structures",
        "Associated Lymphadenopathy"
    ],
    "ILD": [
        "Reticular opacities",
        "Ground-glass opacities throughout the lung fields",
        "Evidence of honeycombing",
        "Traction bronchiectasis",
        "Loss of volume"
    ],
    "Infiltration": [
        "Irregular or fuzzy borders around white areas",
        "Blurring",
        "Hazy or cloudy areas",
        "Increased density or opacity of lung tissue",
        "Air bronchograms",
    ],
    "Lung Lesion": [
        "Solitary pulmonary nodule",
        "Mass or large lesion",
        "Cavitation within lesion",
        "Consolidation",
        "Hilar enlargement or lymphadenopathy",
    ],
    "Lung Opacity": [
        "Consolidation",
        "Ground-glass opacity",
        "Nodules or masses",
        "Interstitial opacities",
        "Pleural effusion"
    ],
    "Nodule": [
        "Calcifications or mineralizations",
        "Shadowing",
        "Distortion or compression of tissues",
        "Anomalous structure or irregularity in shape",
        "Nodular shape that protrudes into a cavity or airway",
    ],
    "Mass": [
        "Distinct edges or borders",
        "Calcifications or speckled areas",
        "Small round oral shaped spots",
        "White shadows"
    ],
    "Pleural Effusion": [
        "Blunting of costophrenic angles",
        "Opacity in the lower lung fields",
        "Mediastinal shift",
        "Reduced lung volume",
        "Meniscus sign or veil-like appearance"
    ],
    "Pleural Thickening": [
        "Thickened pleural line",
        "Loss of sharpness of the mediastinal border",
        "Calcifications on the pleura",
        "Lobulated peripheral shadowing",
        "Loss of lung volume",
    ],
    "Pleural Other": [
        "Thickened pleural line",
        "Loss of sharpness of the mediastinal border",
        "Calcifications on the pleura",
        "Lobulated peripheral shadowing",
        "Loss of lung volume",
    ],
    "Pneumonia": [
        "Consolidation of lung tissue",
        "Air bronchograms",
        "Cavitation",
        "Interstitial opacities"
    ],
    "Pneumothorax": [
        "Tracheal deviation",
        "Deep sulcus sign",
        "Increased radiolucency",
        "Flattening of the hemidiaphragm",
        "Absence of lung markings",
        "Shifting of the mediastinum"
    ],
    "Pulmonary fibrosis": [
        "Reticular shadowing of the lung peripheries",
        "Volume loss",
        "Thickened and irregular interstitial markings",
        "Bronchial dilation",
        "Shaggy heart borders"
    ],
    "Scoliosis": [
        "Lateral Curvature of the Spine",
        "Vertebral Rotation",
        "Asymmetry of the Rib Cage",
        "Uneven Shoulders or Pelvis",
        "Changes in the Space Between Ribs and Pelvis"
    ],
    "Support Devices": [
        "Endotracheal tubes",
        "Central venous catheters",
        "Pacemakers and defibrillator leads",
        "Chest tubes",
    ],
    "Tube": [
        "Endotracheal Tube Placement",
        "Chest Tube Position",
        "Central Venous Catheters",
        "Feeding Tubes",
        "Pacemaker Leads"
    ],
    "Tuberculosis": [
        "Upper Lobe Infiltrates",
        "Cavitary Lesions",
        "Nodal Enlargement",
        "Miliary Pattern",
        "Pleural Effusion"
    ],
    "COVID19": [
        "Bilateral Multifocal Ground-Glass Opacities",
        "Consolidation",
        "Peripheral Distribution",
        "Crazy-Paving Pattern",
        "Linear Opacities"
    ]
}

custom_mapping = {
    # NIH mappings
    "Effusion": "Pleural Effusion",
    "Pleural_Thickening": "Pleural Thickening",
    # "Fibrosis": "Pulmonary fibrosis",
    # OpenI mappings
    "Lesion": "Lung Lesion",
    "Opacity": "Lung Opacity",
}


CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}


def generate_class_prompts(class_names: str, mode="pos", prompt_file=None,
                           prompt_style: str = "xplainer"):
    assert mode in ["pos", "neg"], "mode must be either pos or neg"
    assert prompt_style in ["biovil", "xplainer", "chexzero", "gloria"]
    if prompt_file is not None:
        with open(prompt_file, "r") as f:
            prompt_dict = json.load(f)
    else:
        prompt_dict = predefined_prompt_dict

    sampled_prompts = {}
    for pathology in class_names:

        # For custom mappings
        if pathology in custom_mapping:
            pathology = custom_mapping[pathology]

        cur_prompts = []
        if prompt_style == "xplainer":
            for obs in prompt_dict[pathology]:
                if mode == "pos":
                    prompt = f"There are {obs.lower()} indicating {pathology.lower()}."
                elif mode == "neg":
                    prompt = f"There are no {obs.lower()} indicating {pathology.lower()}."
                cur_prompts.append(prompt)

        elif prompt_style == "biovil":
            if mode == "pos":
                prompt = f"Findings suggesting {pathology.lower()}."
            elif mode == "neg":
                prompt = f"No evidence of {pathology.lower()}."
            cur_prompts.append(prompt)
        elif prompt_style == "chexzero":
            if mode == "pos":
                prompt = f"{pathology.lower()}."
            elif mode == "neg":
                prompt = f"No {pathology.lower()}."
            cur_prompts.append(prompt)
        elif prompt_style == "gloria":
            v = CHEXPERT_CLASS_PROMPTS[pathology]
            keys = list(v.keys())
            # severity
            for k0 in v[keys[0]]:
                # subtype
                for k1 in v[keys[1]]:
                    # location
                    for k2 in v[keys[2]]:
                        cur_prompts.append(f"{k0} {k1} {k2}")

            if len(cur_prompts) > 5:
                cur_prompts = random.sample(cur_prompts, 5)

        sampled_prompts[pathology] = cur_prompts

    return sampled_prompts


if __name__ == "__main__":
    class_names = ["Atelectasis", "Cardiomegaly",
                   "Consolidation", "Edema", "Pleural Effusion"]
    prompts = generate_class_prompts(
        class_names, mode="neg", prompt_style="biovil")
    from pprint import pprint
    pprint(prompts)
