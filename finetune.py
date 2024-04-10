from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import MusicgenForConditionalGeneration, Trainer, TrainingArguments, MusicgenProcessor, AutoProcessor, AutoTokenizer, MusicgenDecoderConfig
import os, json, scipy, torchaudio, torch
from torch.optim import Adam
from audiocraft.data.music_dataset import MusicDataset


processor = AutoProcessor.from_pretrained('facebook/musicgen-medium')
#Encodec_processor = AutoProcessor.from_pretrained('facebook/encodec_32khz')
musicgen_config = MusicgenDecoderConfig.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained('facebook/musicgen-medium')
#Encodec_model = AutoModel.from_pretrained('facebook/encodec_32khz')
tokenizer = AutoTokenizer.from_pretrained('facebook/musicgen-medium')


def convert_audio_channels(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape

    return wav.mean(dim=-2)

class mytraindataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        path = 'dataset/train/'
        files = os.listdir(path)
        json_files = [file for file in files if file.endswith('.json')]
        file_names = [file[: -5] for file in json_files]
        self.data = []
        for name in file_names:
            json_path = path + name + '.json'
            wav_path = path + name + '.wav'
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
            audio, sr = torchaudio.load(wav_path)
            audio = convert_audio_channels(audio) 
            data["audio"] = audio
            self.data.append(data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
class myvaliddataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        path = 'dataset/valid/'
        files = os.listdir(path)
        json_files = [file for file in files if file.endswith('.json')]
        file_names = [file[: -5] for file in json_files]
        self.data = []
        for name in file_names:
            json_path = path + name + '.json'
            wav_path = path + name + '.wav'
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
            audio, sr = torchaudio.load(wav_path)
            audio = convert_audio_channels(audio) 
            data["audio"] = audio
            self.data.append(data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class mydebugdataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        path = 'dataset/debug/'
        files = os.listdir(path)
        json_files = [file for file in files if file.endswith('.json')]
        file_names = [file[: -5] for file in json_files]
        self.data = []
        for name in file_names:
            json_path = path + name + '.json'
            wav_path = path + name + '.wav'
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
            audio, sr = torchaudio.load(wav_path)
            audio = convert_audio_channels(audio) 
            data["audio"] = audio
            self.data.append(data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


#train_dataset = mytraindataset()
#eval_dataset = myvaliddataset()
debug_dataset = mydebugdataset()

def collate_func(batch):
    texts, audios = [], []
    for item in batch:
        texts.append(item["description"])
        audios.append(item["audio"].numpy())
        inputs = processor(audio=audios, sampling_rate=32000, text=texts, padding=True, return_tensors="pt")
    return inputs

trainloader = DataLoader(debug_dataset, batch_size=2, shuffle=True, collate_fn=collate_func)
validloader = DataLoader(debug_dataset, batch_size=2, shuffle=False, collate_fn=collate_func)

audio=debug_dataset[0]["audio"][:len(debug_dataset[0]["audio"])//8]

inputs = processor(
    text=["80s pop track with bassy drums and synth"],
    audio=audio,
    padding=True,
    return_tensors="pt",
)

pad_token_id = model.generation_config.pad_token_id
decoder_input_ids = (
    torch.ones((inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long)
    * pad_token_id
)

logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits
loss = model(**inputs, decoder_input_ids=decoder_input_ids, labels=inputs["input_ids"]).loss
print(logits)
print(loss)

'''
if torch.cuda.is_available():
    model = model.cuda()

optimizer = Adam(model.parameters(), lr=1e-4)

def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1    

train()
print("done")
'''
