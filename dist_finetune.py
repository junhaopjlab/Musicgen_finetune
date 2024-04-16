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

class MyAudioDataset(Dataset):
    def __init__(self, data_dir, no_label=False):
        self.data_dir = data_dir
        self.data_map = []

        condition_datadir = 'MFA_corpus/'
        label_datadir = 'split_10s/'

        artist_files = os.listdir(data_dir + '/' + label_datadir)
        for artist in artist_files:
            song_files = os.listdir(data_dir + '/' + label_datadir + artist)
            for song_file in song_files:
                song_splitfiles = os.listdir(data_dir + '/' + label_datadir + artist + '/' + song_file)
                song_splitfiles = [file for file in song_splitfiles if file.endswith(".mp3")]
                names = [file[:-4] for file in song_splitfiles]
                for name in names:
                    self.data_map.append(
                        {
                            "audio": data_dir + '/' + condition_datadir + artist + '/' + song_file + '/' + name + '.wav',
                            "label": data_dir + '/' + label_datadir + artist + '/' + song_file + '/' + name + '.mp3'
                        }
                    )

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data["audio"]
        label = data["label"]

        return audio, label
        
def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans


def preprocess_audio(audio_path, model: MusicGen, duration: int = 10):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[1] < model.sample_rate * duration:
        return None
    end_sample = int(model.sample_rate * duration)
    start_sample = 0
    wav = wav[:, start_sample : start_sample + end_sample]

    assert wav.shape[0] == 1

    wav = wav.cuda()
    wav = wav.unsqueeze(1)

    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)

    codes, scale = gen_audio

    assert scale is None

    return codes

def preprocess_melody_wavs(audio_path, model:MusicGen, duration: int = 10):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[1] < model.sample_rate * duration:
        return None
    end_sample = int(model.sample_rate * duration)
    start_sample = 0
    wav = wav[:, start_sample : start_sample + end_sample]
    return wav

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


    dataset = MyAudioDataset(dataset_path, no_label=no_label)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)


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
        for batch_idx, (audio, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            all_codes = []
            melody_wavs = []
            texts = []

            # where audio and label are just paths
            for melody, l in zip(audio, label):
                inner_audio = preprocess_audio(l, model)  # returns tensor
                melody = preprocess_melody_wavs(melody, model)

                if inner_audio is None:
                    continue

                if use_cfg:
                    codes = torch.cat([inner_audio, inner_audio], dim=0)
                else:
                    codes = inner_audio

                all_codes.append(codes)
                melody_wavs.append(melody)
                texts.append("")

            attributes, _ = model._prepare_tokens_and_attributes(texts, prompt=None, melody_wavs=melody_wavs)
            conditions = attributes
            if use_cfg:
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions = conditions + null_conditions
            tokenized = model.lm.condition_provider.tokenize(conditions)
            cfg_conditions = model.lm.condition_provider(tokenized)
            condition_tensors = cfg_conditions

            if len(all_codes) == 0:
                continue

            codes = torch.cat(all_codes, dim=0)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output = model.lm.compute_predictions(
                    codes=codes, conditions=[], condition_tensors=condition_tensors
                )

                codes = codes[0]
                logits = lm_output.logits[0]
                mask = lm_output.mask[0]

                codes = one_hot_encode(codes, num_classes=2048)

                codes = codes.cuda()
                logits = logits.cuda()
                mask = mask.cuda()

                mask = mask.view(-1)
                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes.view(-1, 2048)[mask]

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
    


