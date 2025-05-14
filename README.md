# lora2wildcard

This is a small script that generates a wildcard from the lora directory of A1111 or reforge.  
Extract the tags used for training from the metadata in safetensors and use them in wildcard.  
If a json file output by civitai helper exists, it is also used.  

### usage
```sh
python lora2wildcard.py LORA_DIR_PATH

python lora2wildcard.py -h
```

### sample output
```sh
<lora:2b-s1-illustrious-lora-nochekaiser:0.8>,1girl, 2b, black blindfold, black dress, black hairband, blindfold, clothing cutout, dress, hairband, juliet sleeves, long sleeves, mole, mole under mouth, puffy sleeves, short hair, solo, white hair, yorha no. 2 type b
<lora:JN_Asuka_Soryu_Langley_Illus:0.8>,asuka soryu langley, solo
```
