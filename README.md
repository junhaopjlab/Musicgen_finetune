# Musicgen_finetune
About how to finetune MusicGen

# dataset
数据集里存放了用来微调的数据

# finetune
finetune.py文件给出了尝试微调MusicGen的代码。

里面数据集内容为一个纯人声旋律(audio)对应带伴奏歌曲片段(label)
希望通过audio的纯人声condition生成带伴奏的歌曲。
