
import argparse
import json
import os
import random
import time
from pathlib import Path
from datetime import datetime

from safetensors import safe_open
from safetensors.torch import load_file



def extract_metadata_from_safetensors(file_path):
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
        
        if metadata is None:
            print(f"metadata not found")
            return None
        
        return metadata
    except Exception as e:
        print(f"{file_path}:{e}")
        return None

def parse_tags(metadata):
    tags = []
    
    possible_keys = ["ss_tag_frequency", "ss_tag_frequency_0", "tag_frequency", "tags"]
    
    for key in possible_keys:
        if key in metadata:
            try:
                tag_data = json.loads(metadata[key])
                
                if isinstance(tag_data, dict):
                    # {tag : freq}
                    for tag, freq in tag_data.items():
                        if isinstance(freq, dict):
                            for _tag, _freq in freq.items():
                                tags.append({"tag": _tag, "frequency": _freq})
                                #print(f"{_tag = } {_freq = }")
                        else:
                            tags.append({"tag": tag, "frequency": freq})
                            #print(f"{tag = } {freq = }")
                elif isinstance(tag_data, list):
                    # tag list
                    for item in tag_data:
                        if isinstance(item, str):
                            tags.append({"tag": item, "frequency": 1})
                        elif isinstance(item, dict) and "name" in item:
                            tags.append({"tag": item["name"], "frequency": item.get("frequency", 1)})
            except json.JSONDecodeError:
                print("json.JSONDecodeError")
                tags.append({"tag": metadata[key], "frequency": 1})
    
    if "ss_character_tags" in metadata:
        try:
            char_tags = json.loads(metadata["ss_character_tags"])
            if isinstance(char_tags, list):
                for tag in char_tags:
                    tags.append({"tag": tag, "frequency": 1, "type": "character"})
        except json.JSONDecodeError:
            pass
    
    if "ss_dataset_name" in metadata:
        tags.append({"tag": f"dataset: {metadata['ss_dataset_name']}", "frequency": 1, "type": "dataset"})
    
    if "ss_network_args" in metadata:
        try:
            network_args = json.loads(metadata["ss_network_args"])
            if "network_module" in network_args:
                tags.append({"tag": f"network: {network_args['network_module']}", "frequency": 1, "type": "network"})
        except json.JSONDecodeError:
            pass
    
    return tags


def generate_prompt_from_tags(tags, th = -1, prohibited_tags=[]):
    max_count = None
    res = []
    for tag, count in tags:
        #print(f"{tag=} {count=}")
        if not max_count:
            max_count = count

        if tag in prohibited_tags:
            print(f"ignore {tag=}")
            continue

        if th < 0:
            v = random.random() * max_count
        else:
            v = th * max_count
            
        if count > v:
            if False:
                for x in "({[]})":
                    tag = tag.replace(x, '\\' + x)
            res.append(tag)
            
    res = list(set(res))

    return ", ".join(sorted(res))

def get_prompt_from_metadata(lora_path, th, prohibited_tags):
    metadata = extract_metadata_from_safetensors(lora_path)
    if metadata:
        tags = parse_tags(metadata)
        tags.sort(key=lambda x: x.get("frequency", 0), reverse=True)
        tags = [ (k['tag'], k['frequency']) for k in tags]
        return generate_prompt_from_tags(tags, th, prohibited_tags)
    else:
        print(f"{lora_path=} metadata not found!")
        return ""


def get_activation_text_from_json(lora_path):
    
    lora_path = Path(lora_path)
    
    info_path = lora_path.with_suffix(".json")
    
    if info_path.is_file():
        info = {}
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            return info.get("activation text", "")
    else:
        return ""

def get_safetensors_files(dir_path):
    if not dir_path.is_dir():
        raise ValueError(f"dir not found: {dir_path}")
    
    safetensors_files = dir_path.glob('**/*.safetensors')
    
    return safetensors_files


def get_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_path_token(p):
    parts = p.parts
    if len(parts) >= 2:
        last_two = parts[-2:]
        return '_'.join(last_two)
    elif len(parts) == 1:
        return parts[0]
    else:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Extract tags used for training from lora files and generate wildcards. Omit tags that are used infrequently.")
    parser.add_argument("lora_dir_path", help="lora dir path")
    parser.add_argument("--th", "-t", type=float, default=0.5, help="tag threshold. 0.5 means to use up to half the frequency of the most frequent tag.")
    parser.add_argument("--weight", "-w", type=float, default=1.0, help="lora weight")
    parser.add_argument("--prohibited_tags", "-pt", default="simple background, white background", help="")
    parser.add_argument("--act", "-a", action="store_true", help="Prioritize activation text in .json files than training tags.")
    
    
    start_time = time.time()
    
    args = parser.parse_args()
    
    prohibited_tags = args.prohibited_tags.split(",")
    prohibited_tags = [s.strip() for s in prohibited_tags]
    
    print(f"{prohibited_tags = }")
    
    dir_path = Path(args.lora_dir_path)
    
    safetensors_files = get_safetensors_files(dir_path)
    
    prompt_list = []
    
    for sf in safetensors_files:
        
        if args.act:
            prompt = get_activation_text_from_json(sf)
        else:
            prompt = get_prompt_from_metadata(sf, args.th, prohibited_tags)
        
        if not prompt:
            if args.act:
                prompt = get_prompt_from_metadata(sf, args.th, prohibited_tags)
            else:
                prompt = get_activation_text_from_json(sf)
        
        prompt = f"<lora:{Path(sf).stem}:{args.weight:.2f}>" + "," + prompt
        
        prompt_list.append(prompt)
    
    output_path = get_path_token(dir_path) + "_" + get_time_str() + ".txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(prompt_list))
    
    end_time = time.time()
    
    print(f"elapsed time : {end_time-start_time}")
    
    return 0

if __name__ == "__main__":
    exit(main())