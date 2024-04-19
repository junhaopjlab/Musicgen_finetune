import torchaudio
from audiocraft.models import MusicGen
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import wandb
import argparse
import typing as tp
from typing import Tuple, List
from torch.utils.data import Dataset
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
from torch.nn import functional as F
import datetime


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
        for audio_id in audio_ids_list:
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
    

vocal_paths = [
    '/mnt/petrelfs/liuzihan/music_datasets/processed_data/wandhiSongs/API/张韶涵/淋雨一直走_张韶涵_1/2.wav',
    '/mnt/petrelfs/liuzihan/music_datasets/processed_data/wandhiEN_test/MFA_corpus/Adele/Hello/2.wav',
    '/mnt/petrelfs/liuzihan/music_datasets/processed_data/wandhiEN_test/ACE/TaylorSwift/You Belong With Me - Taylor Swift/10.wav'

]

def gen_wav(vocal_paths, model, current_step, eval_table):    
    audios = []
    for p in vocal_paths:
        melody, sr = torchaudio.load(p)
        wav = model.generate_with_chroma([''], melody[None], sr)
        wav = wav[0].cpu().numpy().transpose(1,0) #soundfile (num_samples, 1 channel)
        audios.append(wandb.Audio(wav, sample_rate=sr))
    eval_table.add_data(current_step, audios[0], audios[1], audios[2])
    # return eval_table
 

        
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



def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)

    return result


# def one_hot_encode(tensor, num_classes=2048):
#     shape = tensor.shape
#     one_hot = torch.zeros((shape[0], shape[1], num_classes))

#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             index = tensor[i, j].item()
#             one_hot[i, j, index] = 1

#     return one_hot

# def one_hot_encode_4d(tensor, num_classes=2048):
#     #[batch_size, num_codebooks, frames] -> [batch_size, num_codebooks, frames, num_classes]
#     shape = tensor.shape
#     one_hot = torch.zeros((shape[0], shape[1], shape[2], num_classes))

#     for b in range(shape[0]):
#         for i in range(shape[1]):
#             for j in range(shape[2]):
#                 index = tensor[b, i, j].item()
#                 one_hot[b, i, j, index] = 1
#     return one_hot


def compute_cross_entropy(
       logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    )-> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

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
    warmup_steps: int = 16,
    batch_size: int = 32,
    use_cfg: bool = False,
    num_workers: int = 1,
):
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
    torch.cuda.set_device(local_rank)
    
    if local_rank == 0:
        if use_wandb:
            wandb.login()
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            run = wandb.init(
                project="audiocraft_musicgen_melody",
                name = nowtime,
                config={
                    'model_id': model_id,
                    'dataset': dataset_path,
                    'lr': lr,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'warmup_steps': warmup_steps
                })
            columns=["id", "gen_audio1", "gen_audio2", "gen_audio3"]
            eval_table = wandb.Table(columns=columns)
            
   
   
    
    model = MusicGen.get_pretrained(model_id)
    model.lm = model.lm.to(torch.float32)  # important

    model.lm = model.lm.to(local_rank)
    model.lm = DDP(model.lm, device_ids=[local_rank], output_device=local_rank)
    model.lm = model.lm.module 


    dataset = MyAudioDataset(args.dataset_path, model.sample_rate)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,collate_fn=my_collate_fn)


    learning_rate = lr
    model.lm.train()

    scaler = torch.cuda.amp.GradScaler()
    if local_rank == 0:
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

    # criterion = nn.CrossEntropyLoss().to(local_rank)


    num_epochs = epochs

    save_step = save_step
    save_models = False if save_step is None else True

    save_path = os.path.join(args.save_path, 'models')

    os.makedirs(save_path, exist_ok=True)

    # if local_rank == 0:
    #     writer = SynmmaryWriter(log_dir=os.path.join(args.save_path, 'log'))
    #     os.makedirs(os.path.join(args.save_path, 'log'), exist_ok=True)

    current_step = 0

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
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
                logits = lm_output.logits
                mask = lm_output.mask
                ce, ce_per_codebook = compute_cross_entropy(logits, codes, mask)
                loss = ce

                # codes = one_hot_encode_4d(codes, num_classes=2048) #[batch_size, num_codebooks, frames, num_classes]
                # codes = codes.cuda()
                # logits = lm_output.logits.cuda() #[batch_size, num_codebooks, frames, num_classes]
                # mask = lm_output.mask.cuda()#[batch_size, num_codebooks, frames]

                # mask = mask.reshape(-1)
                # masked_logits = logits.reshape(-1, 2048)[mask]
                # masked_codes = codes.reshape(-1, 2048)[mask]

                # loss = criterion(masked_logits, masked_codes)

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

            
            if local_rank == 0:
                print(
                f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, current_step: {current_step}, Loss: {loss.item()}, lr:{optimizer.param_groups[0]['lr']}"#, loss_ce_per_codebook:{ce_per_codebook.item()}
                )
                if use_wandb:
                    run.log(
                        {
                            "loss": loss.item(),
                            "lr": optimizer.param_groups[0]['lr'], #optimizer.param_groups[0]['lr'], scheduler.get_last_lr()[0]
                            "total_norm": total_norm,
                            "epoch": epoch,
                            'batch_idx': batch_idx,
                            "step": current_step,
                            "loss_ce1": ce_per_codebook[0].item(),
                            "loss_ce2": ce_per_codebook[1].item(),
                            "loss_ce3": ce_per_codebook[2].item(),
                            "loss_ce4": ce_per_codebook[3].item(),
                        }
                    )
            # print(
            #     f"(rank = {rank}, local_rank = {local_rank}) Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}"
            # )

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
            if local_rank == 0:
                print(current_step, save_step, save_models,  int(current_step) % save_step == 0)
            if local_rank == 0 and save_models:
                if (
                    current_step == int(current_step)
                    and int(current_step) % save_step == 0
                ):
                    torch.save(
                        model.lm.state_dict(), f"{save_path}/lm_{current_step}.pt"
                    )
            
                    if use_wandb:  # eval
                        model.lm.eval()
                        with torch.no_grad():                            
                            gen_wav(vocal_paths, model, current_step, eval_table)                         
                        model.lm.train()


        
    if local_rank == 0:
        torch.save(model.lm.state_dict(), f"{save_path}/lm_final.pt")
        if use_wandb:
            run.log({"eval_gen_audio": eval_table})   
            run.finish()


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
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=False, default='facebook/musicgen-melody')
    parser.add_argument('--lr', type=float, required=False, default=1e-5)
    parser.add_argument('--epochs', type=int, required=False, default=2)
    parser.add_argument('--use_wandb', type=int, required=False, default=1)
    parser.add_argument('--save_step', type=int, required=False, default=None)
    parser.add_argument('--no_label', type=int, required=False, default=0)
    parser.add_argument('--tune_text', type=int, required=False, default=0)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-5)
    parser.add_argument('--grad_acc', type=int, required=False, default=2)
    parser.add_argument('--warmup_steps', type=int, required=False, default=2)
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--use_cfg', type=int, required=False, default=0)
    parser.add_argument('--num_workers', type=int, required=False, default=2)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank ==0:
        print('args:', args)

    main(args)
    


