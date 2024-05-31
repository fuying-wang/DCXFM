import torch
import torch.nn as nn
import os
import json
import yaml
from dcxfm.third_party.medklip.models.model_MedKLIP import MedKLIP
from dcxfm.third_party.medklip.models.tokenization_bert import BertTokenizer
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

original_class = [
    'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
    'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
    'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
    'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
    'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
    'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
    'tail_abnorm_obs', 'excluded_obs', 'covid19'
]


def get_tokenizer(tokenizer, target_text):
    target_tokenizer = tokenizer(list(target_text),
                                 padding='max_length',
                                 truncation=True,
                                 max_length=64,
                                 return_tensors="pt")

    return target_tokenizer


def load_pretrained_medklip(model_path="/home/fywang/Documents/CXRSeg/pretrained/MedKLIP",
                            device=torch.device("cuda:0")):

    config = yaml.load(open(os.path.join(BASE_DIR, "configs/MedKLIP_config.yaml"), 'r'),
                       Loader=yaml.Loader)

    json_book = json.load(open(config['disease_book'], 'r'))
    disease_book = [json_book[i] for i in json_book]
    ana_book = ['It is located at ' + i for i in ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
                                                  'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
                                                  'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
                                                  'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
                                                  'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
                                                  'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
                                                  'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
                                                  'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other']]
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    ana_book_tokenizer = get_tokenizer(tokenizer, ana_book).to(device)
    disease_book_tokenizer = get_tokenizer(tokenizer, disease_book).to(device)

    print("Creating model")
    model = MedKLIP(config, ana_book_tokenizer,
                    disease_book_tokenizer, mode='train').to(device)
    model = nn.DataParallel(model, [0])
    checkpoint = torch.load(os.path.join(
        model_path, "checkpoint_final.pth"), map_location='cpu')
    state_dict = checkpoint['model']
    msg = model.load_state_dict(state_dict)
    print(msg)
    print('load checkpoint from %s' %
          os.path.join(model_path, "checkpoint_final.pth"))
    return model


if __name__ == "__main__":
    model = load_pretrained_medklip()
    print(model)
