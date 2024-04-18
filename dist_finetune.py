import torchaudio
from audiocraft.models import MusicGen
from transformers import get_scheduler
import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
# import wandb
import argparse

from torch.utils.data import Dataset

from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout

import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


DATA_DIR = '/mnt/petrelfs/liuzihan/music_datasets/processed_data/wandhiSongs' #'/mnt/petrelfs/share_data/liuzihan/acc_dataset_zh/'
# audio_ids_path = os.path.join( DATA_DIR, 'all_audio_ids.txt')
# input_dir = os.path.join( DATA_DIR, 'API')
# label_dir = os.path.join( DATA_DIR, 'split_10s')


class MyAudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate):
        self.data_map = []
        self.sample_rate = sample_rate
        audio_ids_path = os.path.join( DATA_DIR, 'all_audio_ids.txt')
        input_dir = os.path.join( DATA_DIR, 'API')
        label_dir = os.path.join( DATA_DIR, 'split_10s')
        with open(audio_ids_path, 'r', encoding='utf-8') as file:
            audio_ids_list = [line.strip() for line in file.readlines()]
        for audio_id in audio_ids_list[:100]:
            input_path = os.path.join(input_dir, audio_id+'.wav')
            label_path = os.path.join(label_dir, audio_id +'.mp3')
            if os.path.exists(input_path) and os.path.exists(label_path): 
                 self.data_map.append({
                    'audio_input':input_path,
                    'audio_label': label_path
                 }
                 )

    def __len__(self):
        return len(self.data_map)
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data["audio_input"]
        label = data["audio_label"]
        return audio, label , self.sample_rate    



# descriptions tp.Sequence[tp.Optional[str]]
# input_wavs torch.Tensor [B,C,T]
# label_wavs torch.Tensor [B,C,T]
def my_collate_fn(batch):
    descriptions = []
    input_wavs = []
    label_wavs  = []
    for (audio, label, sr) in batch:
        audio_input = resample_audio(audio, sr)
        audio_label = resample_audio(label, sr)
        if audio_input is None or audio_label is None:
            continue
        descriptions.append('')
        input_wavs.append(audio_input)
        label_wavs.append(audio_label)
    input_wavs = torch.stack(input_wavs, dim=0)
    label_wavs = torch.stack(label_wavs, dim=0)
    return descriptions, input_wavs, label_wavs
    







    


# class AudioDataset(Dataset):
#     def __init__(self, data_path, use_cfg =False, text_label= False):
#         self.data = torch.load(data_path)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         audio_id = sample['audio_id']
#         audio_input = resample_audio(sample['audio_input'], sample['sr_input'])
#         audio_label = resample_audio(sample['audio_label'], sample['sr_label'])
#         return {'audio_id':audio_id, 'audio_input':audio_input, 'audio_label':audio_label}

    

        
def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans


def resample_audio(audio_path, sample_rate, duration: int = 10):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(dim=0, keepdim=True) #单声道
    start_sample = 0
    end_sample = int(sample_rate * duration)
    if wav.shape[1] <  end_sample:
        return None
    wav = wav[:, start_sample : start_sample + end_sample]
    return wav

    


# def preprocess_audio(audio_path, model: MusicGen, duration: int = 10):
#     wav, sr = torchaudio.load(audio_path)
#     wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
#     wav = wav.mean(dim=0, keepdim=True)
#     if wav.shape[1] < model.sample_rate * duration:
#         return None
#     end_sample = int(model.sample_rate * duration)
#     start_sample = 0
#     wav = wav[:, start_sample : start_sample + end_sample]

#     assert wav.shape[0] == 1

#     wav = wav.cuda()
#     wav = wav.unsqueeze(1)

#     with torch.no_grad():
#         gen_audio = model.compression_model.encode(wav)

#     codes, scale = gen_audio

#     assert scale is None

#     return codes

# def preprocess_melody_wavs(audio_path, model:MusicGen, duration: int = 10):
#     wav, sr = torchaudio.load(audio_path)
#     wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
#     wav = wav.mean(dim=0, keepdim=True)
#     if wav.shape[1] < model.sample_rate * duration:
#         return None
#     end_sample = int(model.sample_rate * duration)
#     start_sample = 0
#     wav = wav[:, start_sample : start_sample + end_sample]
#     return wav

def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)

    return result


def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes))

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1

    return one_hot

def one_hot_encode_4d(tensor, num_classes=2048):
    #[batch_size, num_codebooks, frames] -> [batch_size, num_codebooks, frames, num_classes]
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], shape[2], num_classes))

    for b in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                index = tensor[b, i, j].item()
                one_hot[b, i, j, index] = 1
    return one_hot

def train(
    dataset_path: str,
    model_id: str,
    lr: float,
    epochs: int,
    use_wandb: bool,
    no_label: bool = False,
    tune_text: bool = False,
    save_step: int = 100,
    grad_acc: int = 8,
    use_scaler: bool = False,
    weight_decay: float = 1e-5,
    warmup_steps: int = 4000,
    batch_size: int = 32,
    use_cfg: bool = False,
    num_workers: int = 1,
):
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
    torch.cuda.set_device(local_rank)
    
    # if local_rank == 0 and use_wandb:
    #     run = wandb.init(project="audiocraft")

    
    model = MusicGen.get_pretrained(model_id)
    model.lm = model.lm.to(torch.float32)  # important

    model.lm = model.lm.to(local_rank)
    model.lm = DDP(model.lm, device_ids=[local_rank], output_device=local_rank)
    model.lm = model.lm.module 


    dataset = MyAudioDataset(DATA_DIR, model.sample_rate)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,collate_fn=my_collate_fn)


    learning_rate = lr
    model.lm.train()

    scaler = torch.cuda.amp.GradScaler()

    if tune_text:
        print("Tuning text")
    else:
        print("Tuning everything")

    # from paper
    optimizer = AdamW(
        model.lm.condition_provider.parameters()
        if tune_text
        else model.lm.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        warmup_steps,
        int(epochs * len(train_dataloader) / grad_acc),
    )

    criterion = nn.CrossEntropyLoss().to(local_rank)


    num_epochs = epochs

    save_step = save_step
    save_models = False if save_step is None else True

    save_path = "saved_models/"

    os.makedirs(save_path, exist_ok=True)

    current_step = 0

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for batch_idx, (descriptions, input_wavs, label_wavs) in enumerate(train_dataloader):
            optimizer.zero_grad()
            attributes, _ = model._prepare_tokens_and_attributes(descriptions, prompt=None, melody_wavs=input_wavs)
            conditions = attributes
            # if use_cfg:
            #     null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            #     conditions = conditions + null_conditions
            tokenized = model.lm.condition_provider.tokenize(conditions)
            condition_tensors = model.lm.condition_provider(tokenized)

            #获得label_wav的encodec code_id
            label_wavs = label_wavs.cuda()
            with torch.no_grad():
                codes, scale = model.compression_model.encode(label_wavs) #[batch_size, num_codebooks, frames]
            assert scale is None

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output = model.lm.compute_predictions(
                    codes=codes, conditions=[], condition_tensors=condition_tensors
                )

                codes = one_hot_encode_4d(codes, num_classes=2048) #[batch_size, num_codebooks, frames, num_classes]

                codes = codes.cuda()
                logits = lm_output.logits.cuda() #[batch_size, num_codebooks, frames, num_classes]
                mask = lm_output.mask.cuda()#[batch_size, num_codebooks, frames]

                mask = mask.reshape(-1)
                masked_logits = logits.reshape(-1, 2048)[mask]
                masked_codes = codes.reshape(-1, 2048)[mask]

                loss = criterion(masked_logits, masked_codes)

            current_step += 1 / grad_acc

            # assert count_nans(masked_logits) == 0

            (scaler.scale(loss) if use_scaler else loss).backward()

            total_norm = 0
            for p in model.lm.condition_provider.parameters():
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except AttributeError:
                    pass
            total_norm = total_norm ** (1.0 / 2)

            if use_wandb:
                run.log(
                    {
                        "loss": loss.item(),
                        "total_norm": total_norm,
                    }
                )

            print(
                f"(rank = {rank}, local_rank = {local_rank}) Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}"
            )

            if batch_idx % grad_acc != grad_acc - 1:
                continue

            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 0.5)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if local_rank == 0 and save_models:
                if (
                    current_step == int(current_step)
                    and int(current_step) % save_step == 0
                ):
                    torch.save(
                        model.lm.state_dict(), f"{save_path}/lm_{current_step}.pt"
                    )
    if local_rank == 0:
        torch.save(model.lm.state_dict(), f"{save_path}/lm_final.pt")


def ddp_setup():
    init_process_group(backend="nccl")
    
  


def main(args):   
    ddp_setup()

    train(
        dataset_path=args.dataset_path,
        model_id=args.model_id,
        lr=args.lr,
        epochs=args.epochs,
        use_wandb=args.use_wandb,
        save_step=args.save_step,
        no_label=args.no_label,
        tune_text=args.tune_text,
        weight_decay=args.weight_decay,
        grad_acc=args.grad_acc,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        use_cfg=args.use_cfg,
        num_workers=args.num_workers,
    )

    destroy_process_group()


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=False, default='facebook/musicgen-melody')
    parser.add_argument('--lr', type=float, required=False, default=1e-5)
    parser.add_argument('--epochs', type=int, required=False, default=2)
    parser.add_argument('--use_wandb', type=int, required=False, default=0)
    parser.add_argument('--save_step', type=int, required=False, default=None)
    parser.add_argument('--no_label', type=int, required=False, default=0)
    parser.add_argument('--tune_text', type=int, required=False, default=0)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-5)
    parser.add_argument('--grad_acc', type=int, required=False, default=2)
    parser.add_argument('--warmup_steps', type=int, required=False, default=4000)
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--use_cfg', type=int, required=False, default=0)
    parser.add_argument('--num_workers', type=int, required=False, default=2)
    args = parser.parse_args()

    main(args)
    


