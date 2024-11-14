import time
from tqdm import tqdm  # Not required, but you're welcome to use it.
import argparse
from typing import Callable, Sequence

import torch

import utils
import dataloader
from model import TransformerEncoderDecoder
from blue_score import BLEU_score

try:
    import wandb
except ImportError:
    pass


class TransformerRunner:
    """
    Interface between model and training and inference operations
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        src_vocab_size: int,
        tgt_vocab_size: int,
        padding_idx: int = 2,
    ):
        self.opts = opts

        # dataset info
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.padding_idx = padding_idx

        # model hyper-params
        self.num_layers = opts.encoder_num_hidden_layers
        self.d_model = opts.word_embedding_size
        self.d_ff = opts.transformer_ff_size
        self.num_heads = opts.heads  # default = 4, same used in mh_attn
        self.dropout = opts.encoder_dropout
        self.atten_dropout = opts.attention_dropout
        self.is_pre_layer_norm = not self.opts.with_post_layer_norm

        self.BLEU_score = BLEU_score

        self.model = TransformerEncoderDecoder(
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.padding_idx,
            self.num_layers,
            self.d_model,
            self.d_ff,
            self.num_heads,
            self.dropout,
            self.atten_dropout,
            self.is_pre_layer_norm,
            no_src_pos=opts.no_source_pos,
            no_tgt_pos=opts.no_target_pos,
        )

        for p in self.model.parameters():  # parameter initialization
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        # training and eval hyper-params
        self.accum_iter = getattr(self.opts, "gradient_accumulation", 1)
        self.label_smoothing = 0.0  # 0.1
        self.lr = 1.0
        self.warmup = 300  # 3000 // self.accum_iter
        self.factor = 1.0
        self._opt_d = 512  # 512,  # self.d_model

        if getattr(self.opts, "patience", None):
            self.max_epochs = float("inf")
            self.patience = self.opts.patience
        else:
            self.max_epochs = getattr(self.opts, "epochs", 7)
            self.patience = float("inf")

    def load_model(self):
        state_dict = torch.load(self.opts.model_path)
        self.model.load_state_dict(state_dict)
        del state_dict

    def save_model(self):
        device = next(self.model.parameters()).device
        self.model.cpu()
        with a2_utils.smart_open(self.opts.model_path, "wb") as model_file:
            torch.save(self.model.state_dict(), model_file)
        self.model.to(device)

    def init_visualization(self):
        writer = None
        if self.opts.viz_wandb:
            # View at: https://wandb.ai/<opts.viz_wandb>/csc401-w24-a2
            wandb.init(
                name=f"Train-{type(self.model.decoder).__name__}",
                project="csc401-w24-a2",
                entity=self.opts.viz_wandb,
                sync_tensorboard=(self.opts.viz_tensorboard is not None),
            )
            wandb.config = {
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "batch_size": self.opts.batch_size,
                "source_vocab_size": self.src_vocab_size,
                "target_vocab_size": self.tgt_vocab_size,
            }
            wandb.watch(self.model)
        if self.opts.viz_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(comment="a2_run")  # outputs to ./runs/
        return writer

    def update_visualization(self, writer, cur_epoch, loss, bleu_list, log_str):
        if self.opts.viz_tensorboard:
            writer.add_scalar("Loss/train", loss, cur_epoch)
            for n, bleu in bleu_list:
                writer.add_scalar(f"BLEU{n}/train", bleu, cur_epoch)
            writer.add_text("", log_str, cur_epoch)
        if self.opts.viz_wandb:
            # wandb.log({"loss": loss, "global_step": epoch})
            d = {"loss": loss}
            for n, bleu in bleu_list:
                d[f"BLEU-{n}"] = bleu
            wandb.log(d)

    def finalize_visualization(self, writer):
        if self.opts.viz_tensorboard:
            writer.flush()  # Ensure all pending events have been written to disk
            writer.close()
        if self.opts.viz_wandb:
            wandb.finish()

    def train(
        self,
        train_dataloader: a2_dataloader.HansardDataLoader,
        dev_dataloader: a2_dataloader.HansardDataLoader,
        device: torch.device = torch.device("cpu"),
        n_gram_levels: tuple[int, ...] = (4, 3),
    ):
        """
        Train all epochs
        """
        # optimizer,
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: a2_utils.schedule_rate(
                step, self._opt_d, factor=self.factor, warmup=self.warmup
            ),
        )
        # loss function
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        # put on device
        self.model.to(device)
        criterion.to(device)

        writer = self.init_visualization()

        # current state
        cur_step = 0
        cur_epoch = 1

        best_bleu = 0.0
        num_poor = 0

        start = time.time()
        while cur_epoch <= self.max_epochs and num_poor < self.patience:
            self.model.train()
            print(
                f"[Device:{self.opts.device}] Epoch {cur_epoch} Training ====",
                flush=True,
            )
            loss, num_steps = self.train_for_epoch(
                train_dataloader,
                optimizer,
                scheduler,
                criterion,
                self.opts.device,
                self.accum_iter,
            )
            cur_step += num_steps

            torch.cuda.empty_cache()

            print(
                f"[Device:{self.opts.device}] Epoch {cur_epoch} Validation ====",
                flush=True,
            )
            self.model.eval()
            bleu = None
            if (
                cur_epoch > self.opts.skip_eval
            ):  # skip BLEU computation for the first few epochs until model converges
                with torch.no_grad():
                    bleu = self.compute_average_bleu_over_dataset(
                        self.BLEU_score,
                        dev_dataloader,
                        device=self.opts.device,
                        use_greedy_decoding=True,  # use greedy decoding for speed
                    )

            t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
            if bleu is not None:
                bleu_str = " ".join(
                    [f"BLEU-{n}: {b:.4f}" for n, b in zip(n_gram_levels, bleu)]
                )
            else:
                bleu_str = f"BLEU: skipped until epoch {self.opts.skip_eval + 1}"
                bleu = [0.0] * len(n_gram_levels)  # for visualizer
            log_str = f"Epoch {cur_epoch}: loss={loss}, {bleu_str}, time={t}"
            self.update_visualization(
                writer, cur_epoch, loss, zip(n_gram_levels, bleu), log_str
            )
            #print(log_str)

            if bleu[0] < best_bleu:  # use first n-gram level for early stopping
                num_poor += 1
            else:
                num_poor = 0
                best_bleu = bleu[0]
            cur_epoch += 1
        if cur_epoch > self.max_epochs:
            print(f"Finished {self.max_epochs} epochs")
        else:
            print(f"BLEU did not improve after {self.patience} epochs. Done.")

        self.finalize_visualization(writer)

    @staticmethod
    def train_input_target_split(
        target_tokens: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Split target tokens into input and target for maximum likelihood training (teacher forcing)

        Hint:  It is sometimes helpful to debug the model by making it target the exact sample inputs,
        and then ensuring the model can (extremely) over fit in this setting in an epoch or two.

        target_tokens: torch.Tensor Long, [batch_size, seq_len]
        return: the model inputs [batch_size, seq_len - 1],
            and the training targets [batch_size * (seq_len - 1)] as a flat, contiguous tensor
        """
        model_input = target_tokens[:, :-1]
        # Should remove <eos> and <sos>
        training_target = target_tokens[:, 1:].reshape(-1).contiguous()
        
        return model_input, training_target

    @staticmethod
    def train_step_optimizer_and_scheduler(
        optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR
    ) -> None:
        """
        Step the optimizer, zero out the gradient, and step scheduler
        """
        optimizer.step()
        optimizer.zero_grad()
        # lr scheduler
        scheduler.step()

    def train_for_epoch(
        self,
        dataloader: a2_dataloader.HansardDataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        criterion: torch.nn.CrossEntropyLoss,
        device: torch.device = torch.device("cpu"),
        accum_iter: int = 20,
    ):
        """
        Train a single epoch

        Transformers generally perform better with larger batch sizes,
        so we use gradient accumulation (accum_iter) to simulate larger batch sizes.
        This means that we only update the model parameters every accum_iter batches.
        Remember to scale the loss correctly when backpropagating with gradient accumulation.

           1. For every iteration of the `dataloader`
            which yields triples `source, source_lens, targets`, where
                source, [batch_size, src_seq_len] and padded with sos, eos, and padding tokens
                source_lens, [batch_size]
                targets  [batch_size, tgt_seq_len] and padded with sos, eos, and padding tokens

           2. Sends these to the appropriate device via `.to(device)` etc.
           3. Splits the targets into model (`self.model`) input and criterion targets
           4. Gets the logits via the model
           5. Gets the loss via the criterion
           6. Backpropagate gradients through the model
           Hint: how does the loss get scaled when using gradient accumulation?
           7. Updates the model parameters every `accum_iter` iterations:
            a) step the optimizer
            b) zero the gradients
            c) step the scheduler
            Hint: be careful about the end of the epoch edge case.
           8. Returns the per-token cross entropy (as a float not a tensor),
                num iterations (this is not the same as the number of gradient accumulation steps)

        return: float, int
        """
        # set to train mode
        self.model.train()
        total_loss = 0
        num_iters = 0

        for i, (source, source_lens, targets) in enumerate(dataloader):
            # 2. send everything to device
            source = source.to(device)
            source_lens = source_lens.to(device)
            targets = targets.to(device)

            # 3. 
            model_input, training_target = self.train_input_target_split(targets)
            # training_target already flattened

            # 4. 
            # [bs, seq_len, v_size]
            logits = self.model(source, model_input, normalize_logits=False)
            #print("LOGITS", logits.shape)
            
            # 5. 
            # [bs * seq_len, v_size] flatten over dimensions except v_size
            logits = logits.reshape(-1, logits.size(-1))
            #print(logits.shape)
            # [bs, seq_len, vocab_size] -> [bs * seq_len, vocab_size]
            logits_flat = logits.view(-1, logits.size(-1))
            # [bs, seq_len] -> [bs * seq_len]
            training_target_flat = training_target.view(-1) 
            loss = criterion(logits_flat, training_target_flat) / accum_iter
            
            # 6.
            loss.backward()
            
            # Update total loss (unscaled)
            total_loss += loss.item() * accum_iter
            num_iters += 1

            # 7. udpate only after accum_iter iterations or at end
            if (i + 1) % accum_iter == 0 or (i + 1) == len(dataloader):
                self.train_step_optimizer_and_scheduler(optimizer, scheduler)

        # Return average loss per token
        return total_loss / num_iters, num_iters

    def test(self, dataloader, n_gram_levels: tuple[int, ...] = (4, 3)):
        self.load_model()
        self.model.to(self.opts.device)
        self.model.eval()
        with torch.no_grad():
            bleu = self.compute_average_bleu_over_dataset(
                self.BLEU_score,
                dataloader,
                device=self.opts.device,
                use_greedy_decoding=self.opts.greedy,  # note this is different than the test default
            )
            bleu_str = " ".join(
                [f"BLEU-{n}: {b:.4f}" for n, b in zip(n_gram_levels, bleu)]
            )
        print(f"The average BLEU score over the test set was {bleu_str}")

    def translate(self, input_sentence):
        """
        Translate a single sentence

        This method translates the input sentence from the model's source
        language to the target language.
        1. Tokenize the input sentence.
        2. Convert tokens into ordinal IDs.
        3. Feed the tokenized sentence into the model.
        4. Decode the output of the sentence into a string.

        Hints:
        You will need the following methods/attributs from the dataset.
        Consult :class:`HansardEmptyDataset` for a description of parameters
        and attributes.
          self.dataset.tokenize(input_sentence)
              This method tokenizes the input sentence.  For example:
              >>> self.dataset.tokenize('This is a sentence.')
              ['this', 'is', 'a', 'sentence']
          self.dataset.source_word2id
              A dictionary that maps tokens to ids for the source language.
              For example: `self.dataset.source_word2id['francophone'] -> 5127`
          self.dataset.source_unk_id
              The speical token for unknown input tokens.  Any token in the
              input sentence that isn't present in the source vocabulary should
              be converted to this special token.
          self.dataset.target_id2word
              A dictionary that maps ids to tokens for the target language.
              For example: `self.dataset.target_word2id[6123] -> 'anglophone'`

        return: str
        """
        sos_idx, eos_idx, pad_idx = (
            self.dataset.target_sos_id,
            self.dataset.target_eos_id,
            self.padding_idx,
        )
        
        # 1.
        tokens = self.dataset.tokenize(input_sentence)
        
        # 2. 
        source_tokens = torch.tensor(
            # source_word2id : token -> id
            # place tensor on correct device
            [[self.dataset.source_word2id.get(token, self.dataset.source_unk_id) 
              for token in tokens]], 
            device=self.opts.device)

        # 3.
        if self.opts.greedy:
            hypotheses = self.model.greedy_decode(source_tokens, sos_idx, eos_idx)
        else:
            hypotheses = self.model.beam_search_decode(
                source_tokens, sos_idx, eos_idx, k=self.opts.beam_width
            )

     
        # 4. decode
        output_tokens = []
        for token_id in hypotheses[0]:
            # these tokens should not come up but just in case
            # they should not be outputted
            # Note: in example they allow <sos> keep this way for now
            if token_id not in [eos_idx, pad_idx, sos_idx]:
                output_tokens.append(self.dataset.target_id2word[token_id.item()])

        # sentence
        return " ".join(output_tokens)

    @staticmethod
    def compute_batch_total_bleu(
        bleu_score_func: Callable[[Sequence[str], Sequence[str], int], float],
        target_y_ref: torch.LongTensor,
        target_y_cand: torch.LongTensor,
        sos_idx: int,
        eos_idx: int,
        pad_idx: int,
        n_gram_levels: tuple[int, ...] = (4, 3),
    ) -> tuple[tuple[float, ...], int]:
        """
        Compute the total BLEU score for each n_gram_level in n_gram_levels over elements in a batch.
        Clean up the sequences by removing ALL special tokens (sos_idx, eos_idx, pad_idx).

        Assume that the candidate sequences have been padded after the eos token.

        bleu_score_func: Callable[[Sequence[str], Sequence[str], int], float] from BLEU_score.py
        target_y_ref : torch.LongTensor [batch_size, max_ref_seq_len]
        target_y_cand : torch.LongTensor [batch_size, max_cand_seq_len]
        sos_idx : int start of sentence special token
        eos_idx : int end of sentence special token
        pad_idx : int padding special token
        n_gram_levels : tuple[int] n-gram levels to compute BLEU score at i.e the precisions

        return: list summed BLEU score at each level for batch, batch_size
        """
        batch_size = target_y_ref.size(0)
        bleu = [0.0] * len(n_gram_levels)

        for i in range(batch_size):
            # lists so we can iterate
            ref_seq = target_y_ref[i].tolist()
            cand_seq = target_y_cand[i].tolist()

            # assert len(ref_seq) == len(cand_seq)
            # clean both seq 
            ref_clean = []
            for token in ref_seq:
                if token not in [sos_idx, eos_idx, pad_idx]:
                    ref_clean.append(str(token))
                elif token == eos_idx:
                    break
            cand_clean = []
            for token in cand_seq:
                if token not in [sos_idx, eos_idx, pad_idx]:
                    cand_clean.append(str(token))
                elif token == eos_idx:
                    break

            #blue scores
            for j, n in enumerate(n_gram_levels):
                bleu[j] = bleu_score_func(ref_clean, cand_clean, n)
                #print("LOOP", j, bleu[j])

        return tuple(bleu), batch_size

    def compute_average_bleu_over_dataset(
        self,
        bleu_score_func: Callable[[Sequence[str], Sequence[str], int], float],
        dataloader: a2_dataloader.HansardDataLoader,
        device: torch.device = torch.device("cpu"),
        max_len: int = 100,
        use_greedy_decoding: bool = True,
        n_gram_levels: tuple[int, ...] = (4, 3),
    ) -> tuple[float, ...]:
        """
        Determine the average BLEU score across sequences

        This function computes the average BLEU score across all sequences in
        a single loop through the `dataloader`.

        Returns avg_bleu : float
            The total BLEU score summed over all sequences divided by the number of
            sequences
        """

        assert not self.model.training
        sos_idx, eos_idx, pad_idx = (
            dataloader.target_sos_id,
            dataloader.target_eos_id,
            self.padding_idx,
        )

        total_bleu = [0.0] * len(n_gram_levels)
        num = 0.0
        for i, (source_tokens, _, target_tokens) in enumerate(dataloader):
            source_tokens = source_tokens.to(device)
            # this is `cheating` as it uses the target info,
            # but it makes it faster to compute the bleu score
            max_len_i = min(target_tokens.shape[1] + 5, max_len)
            # print(f"max_len_i: {max_len_i} for batch {i}/ {len(dataloader)}", flush=True)

            if use_greedy_decoding:
                candidates = self.model.greedy_decode(
                    source_tokens, sos_idx, eos_idx, max_len_i
                )
            else:
                candidates = self.model.beam_search_decode(
                    source_tokens, sos_idx, eos_idx, max_len_i, k=self.opts.beam_width
                )
            batch_bleu, batch_size = self.compute_batch_total_bleu(
                bleu_score_func, target_tokens, candidates, sos_idx, eos_idx, pad_idx
            )
            for j in range(len(batch_bleu)):
                total_bleu[j] += batch_bleu[j]
            num += batch_size
        #print(total_bleu, num)
        return tuple(b / num for b in total_bleu)
