from pathlib import Path

import torch

from . import tokenizer
from .pretraining import experiment, trainer


def vocab_list(vocab_file: Path):
    tok = tokenizer.load(vocab_file)
    for t, b in tok.decoder.items():
        print(t, b)


def vocab_view(vocab_file: Path, text: str):
    tok = tokenizer.load(vocab_file)
    enc = tok.encode(text)
    dec = (tok.decode([t]) for t in enc)
    print(" ".join(map(str, enc)))
    print(" ".join(dec))


def pretrain_configs():
    for c in experiment.get_configs():
        print(c)


def pretrain_inference(
    vocab_file: Path,
    checkpoint: Path,
    prompt: str,
    max_length: int = 50,
):
    tok = tokenizer.load(vocab_file)
    eos = tok.get_control_token("<|EOS|>")
    chkpt = trainer.load_checkpoint(checkpoint)
    model = trainer.build_model(chkpt.model)
    model.load_state_dict(chkpt.model.state_dict)

    enc = tok.encode(prompt)
    enc = torch.tensor(enc).long().reshape(1, -1)

    output_tokens = list[int]()
    for _ in range(max_length):
        logits = model(enc, inference_mode=True)
        best = torch.argmax(logits, dim=2)
        enc = torch.cat([enc, best], dim=1)

        best_scalar = int(best.item())
        output_tokens.append(best_scalar)
        if best_scalar == eos:
            break

    print(tok.decode(output_tokens))


if __name__ == "__main__":
    import typer

    cli = typer.Typer(add_completion=False)

    cli.command()(vocab_list)
    cli.command()(vocab_view)
    cli.command()(pretrain_configs)
    cli.command()(pretrain_inference)

    cli()
