# Musicgen_finetune
About how to finetune MusicGen
# musicgen
在musicgen文件夹下存放了transformers里关于musicgen的代码。

# finetune
finetune.py文件给出了尝试微调MusicGen的代码。line 134 loss = model(**inputs, decoder_input_ids=decoder_input_ids, labels=inputs["input_ids"]).loss，尝试去得到模型的loss，但报错AttributeError: 'MusicgenConfig' object has no attribute 'vocab_size'。追溯到musicgen/modeling_musicgen.py line 1891-1895.
