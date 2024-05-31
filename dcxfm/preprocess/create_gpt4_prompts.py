import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json
import ipdb

'''
GPT4: 
```
prompt: 
You are a radiology assistant, skilled in providing radiology knowledge. Please describe top 5 most possible observations in chest X-ray images that would occur in a radiology report indicating post radiotherapy changes. Each observation should be one short term. Please anwser the question with the following format: observation1, observation2, observation3, observation4, observation5.


You are a radiology assistant, skilled in providing radiology knowledge. Please find the most possible pathology similar to Emphysema from the given 14 pathologies: "No Finding", "Enlarged Cardiomediastinum","Cardiomegaly","Lung Lesion","Lung Opacity","Edema","Consolidation""Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices".
```
'''

client = OpenAI(
    api_key="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"
)

'''
API-Key:
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
'''

def return_top_5_possible_observations(pathology: str):
    ''' Return top 5 observations from GPT4.''' 

    messages=[
    {"role": "system", 
    "content": "You are a radiology assistant, skilled in providing radiology knowledge."},
    {"role": "user", 
    "content": "Please describe top 5 most possible observations in chest X-ray images that would "\
               f"occur in a radiology report indicating {pathology.lower()}."\
                "Each observation should be one short term."\
                "Please anwser the question with the following format: observation1, observation2, observation3, observation4, observation5"},
    ]
    temperature=0.2
    max_tokens=256
    frequency_penalty=0.0

    response = client.chat.completions.create(
        model="gpt-4",
        messages = messages,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def create_padchest_prompts():
    '''
    Create prompts for the PadChest dataset
    '''
    dataset_dir = "/disk1/fywang/CXR_dataset/preprocessed_csv/PadChest"
    test_df = pd.read_csv(os.path.join(dataset_dir, "padchest_test.csv"))
    pathologies = test_df.columns.tolist()[3:]
    # labels = test_df.values[:, 3:]
    # print(pd.DataFrame(labels, columns=pathologies).describe().T)
    # label_df = test_df.iloc[:, 3:]
    # print(label_df.sum(axis=0).sort_values(ascending=False))
    
    all_observations = dict()
    for pathology in tqdm(pathologies, total=len(pathologies)):
        print(f"Pathology: {pathology}")
        observations = return_top_5_possible_observations(pathology)
        obs_list = observations.split(",")
        all_observations[pathology] = [obs.strip() for obs in obs_list]


    with open(os.path.join(dataset_dir, "padchest_prompts.json"), "w") as f:
        json.dump(all_observations, f, indent=4)


if __name__ == "__main__":
    create_padchest_prompts()