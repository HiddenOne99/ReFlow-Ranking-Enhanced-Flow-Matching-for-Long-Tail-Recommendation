import os
import sys
import math
import time
import random
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = SimpleNamespace(
    train_path='train.inter',
    valid_path='valid.inter',
    test_path='test.inter',

    # ---- Architecture ----
    dims_mlp=[600],
    time_embedding_size=10,
    n_steps=50,
    s_steps=2,
    mlp_dropout=0.4,
    act_func='tanh',
    norm=False,

    # ---- Decode-time popularity adjustment ----
    decode_mode='div',   # 'div'/'sub' keep at div
    decode_gammas=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25],
    decode_floor=1e-6,

    # ---- Ranking ----
    rank_lambda=0.0005,
    rank_num_neg=1000,      # sampled negatives per user
    rank_tau=0.2,          # InfoNCE temperature

    # ---- Optimization ----
    batch_size=4096,          # USERS per batch (autoencoder-style)
    eval_batch_size=2048,     # users per eval chunk
    max_epochs=200,
    patience=10,
    lr=1e-3,
    weight_decay=0.0,

    # ---- Misc ----
    seed=42,
    eval_every=1,
    gpu=0,
    deterministic=True,
    log_path='reflow_log.txt',
)


class FileLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        with open(filename, "w"):
            pass
        self.log = open(filename, "a", buffering=1)
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self, *a, **kw):
        self.terminal.flush()
        self.log.flush()


def resolve_device(cfg_gpu, cli_gpu=None):
    gpu_arg = cli_gpu

    env_val = os.environ.get('CLAP_GPU', None)
    env_gpu = int(env_val) if env_val and env_val.lstrip('-').isdigit() else None

    if gpu_arg is not None: chosen, source = gpu_arg, '--gpu'
    elif env_gpu is not None: chosen, source = env_gpu, 'CLAP_GPU'
    else: chosen, source = int(cfg_gpu), 'CONFIG.gpu'

    if chosen < 0:
        print(f"[device] CPU (forced by {source})")
        return torch.device('cpu')
    if not torch.cuda.is_available():
        print(f"[device] CUDA unavailable; falling back to CPU")
        return torch.device('cpu')
    n = torch.cuda.device_count()
    if chosen >= n:
        print(f"[device] gpu={chosen} but only {n} visible; falling back to 0")
        chosen = 0
    device = torch.device(f'cuda:{chosen}')
    torch.cuda.set_device(device)
    print(f"[device] {device} ({torch.cuda.get_device_name(chosen)}) [src: {source}]")
    return device


def set_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try: torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError: torch.use_deterministic_algorithms(True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        print("[seed] deterministic ON")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_interactions(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0, engine='c',
                     usecols=['user_id', 'item_id'],
                     dtype={'user_id': np.int64, 'item_id': np.int64})
    return df.rename(columns={'user_id': 'user', 'item_id': 'item'})


def build_mappings_from_train(train_df):
    users = sorted(train_df['user'].unique())
    items = sorted(train_df['item'].unique())
    return ({u: i for i, u in enumerate(users)},
            {x: i for i, x in enumerate(items)})


def df_to_csr(df, user_map, item_map, num_users, num_items):
    df = df[df['user'].isin(user_map) & df['item'].isin(item_map)]
    r = df['user'].map(user_map).values
    c = df['item'].map(item_map).values
    d = np.ones(len(df), dtype=np.float32)
    return coo_matrix((d, (r, c)), shape=(num_users, num_items)).tocsr()


def print_eval_block(label, metrics):
    line = (f"  {label}      "
            f"R@10={metrics.get('recall@10', 0):.4f}  N@10={metrics.get('ndcg@10', 0):.4f}  "
            f"R@20={metrics.get('recall@20', 0):.4f}  N@20={metrics.get('ndcg@20', 0):.4f}  "
            f"R@50={metrics.get('recall@50', 0):.4f}  N@50={metrics.get('ndcg@50', 0):.4f}")
    print(line)

def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device) * 2 * math.pi
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def xavier_normal_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)


class FlowModel(nn.Module):
    def __init__(self, dims, time_emb_size, act_func="tanh", norm=False,
                 dropout=0.1, init_dropout=0.0):
        super().__init__()
        self.time_emb_dim = time_emb_size
        self.norm = norm
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        dims = list(dims)
        dims[0] = dims[0] + self.time_emb_dim

        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}[act_func]
        modules = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(d_in, d_out))
            modules.append(act())
        modules.pop()
        self.encoder = nn.Sequential(*modules[:-2]) if len(modules) > 2 else nn.Identity()
        self.head = nn.Sequential(*modules[-2:])

        self.init_dropout = nn.Dropout(init_dropout)
        self.apply(xavier_normal_init)

    def trunk(self, x, t):
        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.init_dropout(x)
        h = torch.cat([x, emb], dim=-1)
        return self.encoder(h)

    def forward(self, x, t):
        return self.head(self.trunk(x, t))

    def forward_features(self, x, t):
        feat = self.trunk(x, t)
        return self.head(feat), feat


class ReFlow(nn.Module):
    def __init__(self, num_users, num_items, train_csr, cfg, device):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_steps = cfg.n_steps
        self.s_steps = cfg.s_steps
        self.device = device
        self.register_buffer(
            "time_steps", torch.linspace(0, 1, self.n_steps + 1)
        )
        in_dim = num_items

        dims = [in_dim] + list(cfg.dims_mlp) + [num_items]
        self.flow_model = FlowModel(
            dims=dims,
            time_emb_size=cfg.time_embedding_size,
            act_func=cfg.act_func,
            norm=cfg.norm,
            dropout=cfg.mlp_dropout,
            init_dropout=0.0,
        )
        item_users = np.asarray((train_csr > 0).sum(axis=0)).ravel().astype(np.float32)
        item_freq = item_users / max(num_users, 1)
        item_freq_t = torch.from_numpy(item_freq)
        self.register_buffer("item_frequencies", item_freq_t)

        self.register_buffer("prior_probs", item_freq_t.clone())

        self.rank_lambda = float(getattr(cfg, 'rank_lambda', 0.0))
        self.rank_num_neg = int(getattr(cfg, 'rank_num_neg', 200))
        self.rank_tau = float(getattr(cfg, 'rank_tau', 0.2))
        if self.rank_lambda > 0:
            print(f"[rank] lambda={self.rank_lambda}  num_neg={self.rank_num_neg}  "
                  f"tau={self.rank_tau}")

    def forward(self, x, t):
        return self.flow_model(x, t)

    def ranking_loss(self, out, x1):
        B, N = out.shape
        max_pos = int((x1 > 0).sum(dim=1).max().item())
        k = min(self.rank_num_neg, max(1, N - max_pos - 1))
        keys = torch.rand(B, N, device=out.device)
        keys.clamp_(1e-12, 1.0).log_().neg_().log_().neg_()
        keys.masked_fill_(x1 > 0, float('-inf'))
        neg_idx = keys.topk(k, dim=1).indices
        s_neg = out.gather(1, neg_idx) / self.rank_tau
        neg_lse = torch.logsumexp(s_neg, dim=1, keepdim=True) - math.log(k)
        s_pos = out / self.rank_tau
        per_item = -s_pos + torch.logaddexp(s_pos, neg_lse.expand_as(s_pos))
        pos_mask = (x1 > 0).float()
        return (per_item * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)

    def flow_loss(self, x1):
        B = x1.size(0)
        steps = torch.randint(0, self.n_steps, (B,), device=x1.device)
        t = self.time_steps[steps].unsqueeze(1)
        x0 = torch.bernoulli(self.prior_probs.expand(B, -1))
        random_mask = torch.rand_like(x1) <= t
        xt = torch.where(random_mask, x1, x0)
        out = self.forward(xt, t.squeeze(-1))
        se = (x1 - out) ** 2
        loss = mean_flat(se).mean()
        if self.rank_lambda > 0:
            loss = loss + self.rank_lambda * self.ranking_loss(out, x1)
        return loss


    @torch.no_grad()
    def predict_scores(self, X_bar):
        Xt = X_bar
        X1_hat = None
        for i_t in range(self.n_steps - self.s_steps, self.n_steps):
            t = self.time_steps[i_t].repeat(Xt.shape[0], 1)
            X1_hat = self.forward(Xt, t.squeeze(-1))
            if i_t == self.n_steps - 1:
                break
            t_next = self.time_steps[i_t + 1].repeat(Xt.shape[0], 1)
            v = (X1_hat - Xt) / (1 - t)
            Xt_pos = Xt + v * (t_next - t)
            Xt_neg = 1 - Xt_pos
            Xt = torch.stack([Xt_neg, Xt_pos], dim=-1).argmax(dim=-1)
            Xt = torch.logical_or(X_bar.to(torch.bool), Xt.to(torch.bool)).float()
        return X1_hat

def metrics_from_scores(scores, test_users, eval_csr, exclude_csrs,
                         num_users, num_items, device,
                         K_list=(5, 10, 20, 50)):
    eval_coo = eval_csr.tocoo()
    test_user_pos_idx = {u: i for i, u in enumerate(test_users)}
    g2t = np.full(num_users, -1, dtype=np.int64)
    for u, i in test_user_pos_idx.items():
        g2t[u] = i
    for csr in exclude_csrs:
        ii = csr.tocoo()
        t_idx = g2t[ii.row]
        keep = t_idx >= 0
        if not keep.any():
            continue
        kept_t = t_idx[keep]; kept_cols = ii.col[keep]
        scores[torch.as_tensor(kept_t, dtype=torch.long, device=device),
               torch.as_tensor(kept_cols, dtype=torch.long, device=device)] = -float('inf')

    maxK = max(K_list)
    _, topk_idx_t = torch.topk(scores, k=maxK, dim=1)
    topk_idx = topk_idx_t.cpu().numpy()
    N_test = len(test_users)

    pos_iids_list = [[] for _ in range(N_test)]
    for u_global, item_global in zip(eval_coo.row, eval_coo.col):
        i = test_user_pos_idx[int(u_global)]
        pos_iids_list[i].append(int(item_global))

    pos_matrix = np.zeros((N_test, num_items), dtype=np.int64)
    for i, pos_iids in enumerate(pos_iids_list):
        if pos_iids:
            pos_matrix[i, pos_iids] = 1
    pos_len_arr = np.sum(pos_matrix, axis=1, keepdims=True)
    pos_idx = pos_matrix[np.arange(N_test)[:, np.newaxis], topk_idx]

    pos_len_safe = np.maximum(pos_len_arr, 1)
    cumhit = np.cumsum(pos_idx, axis=1)
    recall_full = cumhit / pos_len_safe
    recall_per_K = {K: recall_full[:, K-1] for K in K_list}

    len_rank = np.full_like(pos_len_arr, maxK)
    idcg_len = np.minimum(len_rank, pos_len_arr).ravel().astype(np.int64)
    iranks = np.tile(np.arange(1, maxK + 1, dtype=np.float64), (N_test, 1))
    idcg = np.cumsum(1.0 / np.log2(iranks + 1.0), axis=1)
    for row, idx in enumerate(idcg_len):
        if idx > 0:
            idcg[row, idx:] = idcg[row, idx - 1]
        else:
            idcg[row, :] = 1.0
    ranks = np.tile(np.arange(1, maxK + 1, dtype=np.float64), (N_test, 1))
    dcg_per_pos = 1.0 / np.log2(ranks + 1.0)
    dcg = np.where(pos_idx > 0, dcg_per_pos, 0.0)
    dcg_cumsum = np.cumsum(dcg, axis=1)
    ndcg_full = dcg_cumsum / np.maximum(idcg, 1e-12)
    ndcg_per_K = {K: ndcg_full[:, K-1] for K in K_list}

    out = {}
    for K in K_list:
        out[f'recall@{K}'] = float(np.mean(recall_per_K[K]))
        out[f'ndcg@{K}'] = float(np.mean(ndcg_per_K[K]))

    return out


@torch.no_grad()
def reflow_evaluate(model, train_csr, eval_csr, exclude_csrs,
                    num_users, num_items, device,
                    eval_batch_size,
                    K_list=(5, 10, 20, 50), decode_gamma=0.0, decode_mode='div',
                    decode_floor=1e-6):
    eval_coo = eval_csr.tocoo()
    test_users = sorted(set(eval_coo.row.tolist()))
    if not test_users:
        return {}
    model.eval()
    scores = torch.empty((len(test_users), num_items), dtype=torch.float32, device=device)

    adj = None
    if abs(decode_gamma) > 1e-12:
        f = model.item_frequencies.clamp(min=decode_floor)
        if decode_mode == 'sub':
            adj = ('sub', (decode_gamma * torch.log(f)).unsqueeze(0))
        elif decode_mode == 'div':
            adj = ('div', f.pow(decode_gamma).unsqueeze(0))
        else:
            raise ValueError(f"unknown decode_mode '{decode_mode}'")

    for s in range(0, len(test_users), eval_batch_size):
        batch = test_users[s:s + eval_batch_size]
        X_bar = torch.from_numpy(train_csr[batch].toarray()).to(device)
        sc = model.predict_scores(X_bar)
        if adj is not None:
            mode, vec = adj
            if mode == 'sub':
                sc.sub_(vec)
            else:
                row_min = sc.min(dim=1, keepdim=True).values
                sc.sub_(row_min.clamp(max=0.0))
                sc.div_(vec)
        scores[s:s + len(batch)] = sc

    return metrics_from_scores(scores, test_users, eval_csr, exclude_csrs,
                                num_users, num_items, device, K_list)

def train_reflow(args, model, train_csr, valid_csr, test_csr,
                 num_users, num_items, device):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    best_valid = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(num_users)
        num_batches = (num_users + args.batch_size - 1) // args.batch_size
        ep_loss = 0.0

        for b in range(num_batches):
            uidx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            x1 = torch.from_numpy(train_csr[uidx].toarray()).to(device)
            loss = model.flow_loss(x1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().item())

        ep_loss /= max(num_batches, 1)
        elapsed = time.time() - t0
        print(f"=== Epoch {epoch}/{args.max_epochs}  loss={ep_loss:.5f}  ({elapsed:.1f}s) ===")

        if epoch % args.eval_every == 0:
            metrics = reflow_evaluate(
                model, train_csr, valid_csr, [train_csr],
                num_users, num_items, device,
                args.eval_batch_size)
            print_eval_block("valid", metrics)
            valid_score = metrics.get('ndcg@10', 0)
            if valid_score > best_valid:
                best_valid = valid_score
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  [early stop at epoch {epoch}; best epoch was {best_epoch}]")
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    gammas = getattr(args, 'decode_gammas', [0.0])
    mode = getattr(args, 'decode_mode', 'div')

    sel_metric = getattr(args, 'decode_select_metric', 'ndcg@10')
    print(f"\n=== VALID (best epoch {best_epoch})  decode-gamma SELECTION "
          f"(mode={mode}, select on {sel_metric}) ===")
    valid_by_gamma = {}
    for g in gammas:
        vm = reflow_evaluate(
            model, train_csr, valid_csr, [train_csr],
            num_users, num_items, device,
            args.eval_batch_size,
            decode_gamma=g, decode_mode=mode, decode_floor=args.decode_floor)
        valid_by_gamma[g] = vm
        print(f"  [valid {mode} g={g}]  R@20={vm.get('recall@20', 0):.4f}  "
              f"N@20={vm.get('ndcg@20', 0):.4f}  {sel_metric}={vm.get(sel_metric, 0):.4f}")
    g_star = max(gammas, key=lambda g: (valid_by_gamma[g].get(sel_metric, 0.0), -abs(g)))
    print(f"  --> selected gamma* = {g_star}  "
          f"(valid {sel_metric}={valid_by_gamma[g_star].get(sel_metric, 0):.4f})")

    print(f"\n=== TEST (best epoch {best_epoch})  gamma*={g_star} (selected on valid) ===")
    sel = reflow_evaluate(
            model, train_csr, test_csr, [train_csr, valid_csr],
            num_users, num_items, device,
            args.eval_batch_size,
            decode_gamma=g_star, decode_mode=mode, decode_floor=args.decode_floor)
    print_eval_block(f"test g*={g_star}", sel)
    print(f"\n[summary] model=ReFlow  best_epoch={best_epoch}")
    print(f"[summary]   gamma*={g_star}  R@20={sel.get('recall@20', 0):.4f}  "
            f"N@20={sel.get('ndcg@20', 0):.4f}")
    return best_state, {'gamma_star': g_star,
            'valid_by_gamma': valid_by_gamma,
            'test_metrics': sel}

def apply_grid_overrides(cfg):
    blob = os.environ.get('GRID_CONFIG_JSON', None)
    if not blob:
        return cfg
    import json
    try:
        d = json.loads(blob)
    except Exception as e:
        print(f"[grid] parse failed: {e}; ignoring"); return cfg
    for k, v in d.items():
        if not hasattr(cfg, k):
            raise ValueError(f"[grid] unknown CONFIG key '{k}'")
        setattr(cfg, k, v)
    return cfg


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('yes', 'true', 't', 'y', '1'):
        return True
    if s in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got '{v}'")


def build_parser():
    p = argparse.ArgumentParser(
        prog='reflow.py',
        description='ReFlow: Ranking-Enhanced Flow-Matching Recommender.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    d = p.add_argument_group('data')
    d.add_argument('--train_path', type=str, default=None)
    d.add_argument('--valid_path', type=str, default=None)
    d.add_argument('--test_path',  type=str, default=None)

    a = p.add_argument_group('architecture')
    a.add_argument('--dims_mlp', type=int, nargs='+', default=None)
    a.add_argument('--time_embedding_size', type=int, default=None)
    a.add_argument('--n_steps', type=int, default=None)
    a.add_argument('--s_steps', type=int, default=None)
    a.add_argument('--mlp_dropout', type=float, default=None)
    a.add_argument('--act_func', type=str, default=None)
    a.add_argument('--norm', type=str2bool, nargs='?', const=True, default=None)

    g = p.add_argument_group('decode-time popularity adjustment')
    g.add_argument('--decode_mode', type=str, default=None, choices=['div', 'sub'])
    g.add_argument('--decode_gammas', type=float, nargs='+', default=None)
    g.add_argument('--decode_floor', type=float, default=None)

    r = p.add_argument_group('ranking term')
    r.add_argument('--rank_lambda', type=float, default=None)
    r.add_argument('--rank_num_neg', type=int, default=None)
    r.add_argument('--rank_tau', type=float, default=None)

    o = p.add_argument_group('optimization')
    o.add_argument('--batch_size', type=int, default=None)
    o.add_argument('--eval_batch_size', type=int, default=None)
    o.add_argument('--max_epochs', type=int, default=None)
    o.add_argument('--patience', type=int, default=None)
    o.add_argument('--lr', type=float, default=None)
    o.add_argument('--weight_decay', type=float, default=None)

    m = p.add_argument_group('misc')
    m.add_argument('--seed', type=int, default=None)
    m.add_argument('--eval_every', type=int, default=None)
    m.add_argument('--gpu', type=int, default=None)
    m.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=None)
    m.add_argument('--log_path', type=str, default=None)
    return p


def apply_cli_overrides(cfg, ns):
    for k, v in vars(ns).items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise ValueError(f"[cli] unknown CONFIG key '{k}'")
        setattr(cfg, k, v)
    return cfg


def main():
    ns = build_parser().parse_args()
    args = apply_grid_overrides(CONFIG)
    args = apply_cli_overrides(args, ns)

    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    sys.stdout = FileLogger(args.log_path)
    print(f"Config: {vars(args)}")

    device = resolve_device(args.gpu, cli_gpu=ns.gpu)
    set_seed(args.seed, deterministic=args.deterministic)

    train_df = load_interactions(args.train_path)
    valid_df = load_interactions(args.valid_path)
    test_df  = load_interactions(args.test_path)
    user_map, item_map = build_mappings_from_train(train_df)
    num_users, num_items = len(user_map), len(item_map)
    print(f"[data] users={num_users}  items={num_items}  interactions={len(train_df)}")

    train_csr = df_to_csr(train_df, user_map, item_map, num_users, num_items)
    valid_csr = df_to_csr(valid_df, user_map, item_map, num_users, num_items)
    test_csr  = df_to_csr(test_df,  user_map, item_map, num_users, num_items)
    model = ReFlow(num_users, num_items, train_csr, args, device).to(device)
    print(f"[model] ReFlow  dims_mlp={args.dims_mlp}  n_steps={args.n_steps}  "
          f"s_steps={args.s_steps}  rank_lambda={args.rank_lambda}")

    train_reflow(args, model, train_csr, valid_csr, test_csr,
                 num_users, num_items, device)


if __name__ == '__main__':
    main()
