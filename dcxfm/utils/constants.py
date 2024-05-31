localization_tasks = ["Enlarged Cardiomediastinum",
                      "Cardiomegaly",
                      "Lung Lesion",
                      "Airspace Opacity",
                      "Edema",
                      "Consolidation",
                      "Atelectasis",
                      "Pneumothorax",
                      "Pleural Effusion",
                      "Support Devices"]

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

NIH_TASKS = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia"
]

OPENI_TASKS = [
    "Atelectasis",
    "Fibrosis",
    "Pneumonia",
    "Effusion",
    "Lesion",
    "Cardiomegaly",
    "Fracture",
    "Edema",
    "Granuloma",
    "Emphysema",
    "Hernia",
    "Mass",
    "Nodule",
    "Opacity",
    "Infiltration",
    "Pleural_Thickening",
    "Pneumothorax"
]

# 16 classes
PADCHEST_SEEN_CLASSES = [
    # equivalent to chexpert seen classes
    "normal",
    "pneumothorax",
    "pulmonary edema",
    "atelectasis",
    "consolidation",
    "pneumonia",
    "cardiomegaly",
    "pleural effusion",
    "fracture",
    "nodule",
    "pacemaker",
    "pleural thickening",
    "hyperinflated lung",
    "mass",
    "infiltrates",
    "hiatal hernia"
]

# 178 classes
PADCHEST_UNSEEN_CLASSES = [
    'pulmonary fibrosis',
    'chronic changes',
    'kyphosis',
    'pseudonodule',
    'ground glass pattern',
    'unchanged',
    'alveolar pattern',
    'interstitial pattern',
    'laminar atelectasis',
    'apical pleural thickening',
    'suture material',
    'sternotomy',
    'endotracheal tube',
    'heart insufficiency',
    'hemidiaphragm elevation',
    'superior mediastinal enlargement',
    'aortic elongation',
    'scoliosis',
    'sclerotic bone lesion',
    'supra aortic elongation',
    'vertebral degenerative changes',
    'goiter',
    'COPD signs',
    'air trapping',
    'descendent aortic elongation',
    'aortic atheromatosis',
    'metal',
    'hypoexpansion basal',
    'abnormal foreign body',
    'central venous catheter via subclavian vein',
    'central venous catheter',
    'vascular hilar enlargement',
    'vertebral anterior compression',
    'diaphragmatic eventration',
    'calcified densities',
    'fibrotic band',
    'tuberculosis sequelae',
    'volume loss',
    'bronchiectasis',
    'single chamber device',
    'emphysema',
    'vertebral compression',
    'bronchovascular markings',
    'bullas',
    'hilar congestion',
    'exclude',
    'axial hyperostosis',
    'aortic button enlargement',
    'calcified granuloma',
    'clavicle fracture',
    'pulmonary mass',
    'dual chamber device',
    'increased density',
    'surgery neck',
    'osteosynthesis material',
    'costochondral junction hypertrophy',
    'segmental atelectasis',
    'costophrenic angle blunting',
    'calcified pleural thickening',
    'callus rib fracture',
    'mediastinal mass',
    'nipple shadow',
    'surgery heart',
    'pulmonary artery hypertension',
    'central vascular redistribution',
    'tuberculosis',
    'cavitation',
    'granuloma',
    'osteopenia',
    'lobar atelectasis',
    'surgery breast',
    'NSG tube',
    'hilar enlargement',
    'gynecomastia',
    'atypical pneumonia',
    'cervical rib',
    'mediastinal enlargement',
    'major fissure thickening',
    'surgery',
    'azygos lobe',
    'adenopathy',
    'miliary opacities',
    'suboptimal study',
    'dai',
    'mediastinic lipomatosis',
    'surgery lung',
    'mammary prosthesis',
    'humeral fracture',
    'calcified adenopathy',
    'reservoir central venous catheter',
    'vascular redistribution',
    'hypoexpansion',
    'heart valve calcified',
    'pleural mass',
    'loculated pleural effusion',
    'pectum carinatum',
    'subacromial space narrowing',
    'central venous catheter via jugular vein',
    'vertebral fracture',
    'osteoporosis',
    'bone metastasis',
    'lung metastasis',
    'cyst',
    'humeral prosthesis',
    'artificial heart valve',
    'mastectomy',
    'pericardial effusion',
    'lytic bone lesion',
    'subcutaneous emphysema',
    'flattened diaphragm',
    'asbestosis signs',
    'multiple nodules',
    'prosthesis',
    'pulmonary hypertension',
    'soft tissue mass',
    'tracheostomy tube',
    'endoprosthesis',
    'post radiotherapy changes',
    'air bronchogram',
    'pectum excavatum',
    'calcified mediastinal adenopathy',
    'central venous catheter via umbilical vein',
    'thoracic cage deformation',
    'obesity',
    'tracheal shift',
    'external foreign body',
    'atelectasis basal',
    'aortic endoprosthesis',
    'rib fracture',
    'calcified fibroadenoma',
    'reticulonodular interstitial pattern',
    'reticular interstitial pattern',
    'chest drain tube',
    'minor fissure thickening',
    'fissure thickening',
    'hydropneumothorax',
    'breast mass',
    'blastic bone lesion',
    # remove this since it doesn't exist in frontal view
    # 'respiratory distress',
    'azygoesophageal recess shift',
    'ascendent aortic elongation',
    'lung vascular paucity',
    'kerley lines',
    'electrical device',
    'artificial mitral heart valve',
    'artificial aortic heart valve',
    'total atelectasis',
    'non axial articular degenerative changes',
    'pleural plaques',
    'calcified pleural plaques',
    'lymphangitis carcinomatosa',
    'lepidic adenocarcinoma',
    'mediastinal shift',
    'ventriculoperitoneal drain tube',
    'esophagic dilatation',
    'dextrocardia',
    'end on vessel',
    'right sided aortic arch',
    'Chilaiditi sign',
    'aortic aneurysm',
    'loculated fissural effusion',
    'air fluid level',
    'round atelectasis',
    'double J stent',
    'pneumoperitoneo',
    'abscess',
    'pulmonary artery enlargement',
    'bone cement',
    'pneumomediastinum',
    'catheter',
    'surgery humeral',
    'empyema',
    'nephrostomy tube',
    'sternoclavicular junction hypertrophy',
    'pulmonary venous hypertension',
    'gastrostomy tube',
    'lipomatosis'
]