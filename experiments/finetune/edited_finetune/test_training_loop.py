"""Real PyTorch loop/resume test using a tiny fake HF adapter, not a GPU/model test."""
import json
import sys
import tempfile
import types
from pathlib import Path

import polars as pl
import torch

from common import atomic_json, atomic_parquet, code_digest


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, input_ids):
        return types.SimpleNamespace(logits=self.linear(self.dropout(input_ids.float())))

    def gradient_checkpointing_enable(self):
        pass

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = cls()
        model.load_state_dict(torch.load(Path(path) / 'tiny.pt', weights_only=True))
        return model

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path) / 'tiny.pt')


class TinyTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, a, b, **kwargs):
        return {'input_ids': torch.tensor([[len(x) / 10, len(y) / 10] for x, y in zip(a, b)])}

    def save_pretrained(self, path):
        pass


fake = types.ModuleType('transformers')
fake.AutoModelForSequenceClassification = TinyClassifier
fake.AutoTokenizer = TinyTokenizer
sys.modules['transformers'] = fake
import train_score


def execute(*arguments):
    sys.argv = ['train_score.py', *map(str, arguments)]
    train_score.main()


def main():
    with tempfile.TemporaryDirectory(prefix='edited-ft-loop-') as tmp:
        root = Path(tmp)
        initial = root / 'initial'
        initial.mkdir()
        torch.manual_seed(123)
        TinyClassifier().save_pretrained(initial)
        rows = pl.DataFrame({'s1': [f's{i}' for i in range(17)], 'cand': [f'c{i}' for i in range(17)],
            'a': ['short' if i % 2 else 'long business' for i in range(17)],
            'b': ['another business'] * 17, 'y': [i % 2 for i in range(17)], 'weight': [1.] * 17})
        runs = [root / 'uninterrupted', root / 'resumed']
        for folder in runs:
            folder.mkdir()
            atomic_json(folder / 'manifest.json', {'code_sha256': code_digest(), 'checkpoints': [str(initial)] * 2})
            atomic_parquet(rows, folder / 'fit_fold0.parquet')
            atomic_parquet(rows.select('s1', 'cand', 'a', 'b'), folder / 'val_pairs.parquet')
        settings = ['--cpu', '--batch', '4', '--accum', '2', '--checkpoint-every', '1']
        execute('train', '--out', runs[0], *settings)
        real_replace = train_score.os.replace
        def interrupt_after_checkpoint(source, destination):
            real_replace(source, destination)
            if Path(destination).name == 'state.pt':
                raise RuntimeError('Simulated interruption after committed checkpoint')
        train_score.os.replace = interrupt_after_checkpoint
        try:
            execute('train', '--out', runs[1], *settings)
        except RuntimeError as error:
            assert 'Simulated interruption' in str(error)
        else:
            raise AssertionError('Interruption did not fire')
        finally:
            train_score.os.replace = real_replace
        execute('train', '--out', runs[1], *settings)
        execute('train', '--out', runs[1], *settings)
        left = torch.load(runs[0] / 'model_fold0/tiny.pt', weights_only=True)
        right = torch.load(runs[1] / 'model_fold0/tiny.pt', weights_only=True)
        assert all(torch.equal(left[k], right[k]) for k in left)
        assert not (runs[1] / 'model_fold0/state.pt.partial').exists()
        score_args = ['score', '--out', runs[1], '--cpu', '--batch', '4', '--score-chunk', '8']
        execute(*score_args)
        execute(*score_args)
        output = pl.concat([pl.read_parquet(p) for p in sorted((runs[1] / 'val_scores_fold0').glob('*.parquet'))])
        assert output.height == 17 and output['ft_logit'].is_finite().all()
        assert output.select('s1', 'cand').equals(rows.select('s1', 'cand'))
    print('PASS: real PyTorch training, partial accumulation, exact checkpoint resume, sharded inference reuse')


if __name__ == '__main__':
    main()
