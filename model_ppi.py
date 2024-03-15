import argparse
from pathlib import Path

path = Path("/home/new2/huyuecode/sdn2go/data")
onto = 'MF'     # MF,CC,BP
parser = argparse.ArgumentParser()
orgs = 'human'

parser.add_argument("--benchmark_list", type=str, default=path /f"data_{orgs}_{onto}_benchmark.csv")
parser.add_argument("--data_sequence", type=str, default=path /f"data_{orgs}_sequences.csv")
parser.add_argument("--anno_file", type=str, default=path / f"data_{orgs}_annotation.csv")
parser.add_argument("--domain_file", type=str, default=path / f"data_{orgs}_domain.csv")
parser.add_argument("--ppi_file", type=str, default=path / f"data_{orgs}_ppi_1024.csv")

parser.add_argument("--esm_dir", type=str, default=path /f"{orgs}_esm_dir")

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--onto", type=str, default=onto)
parser.add_argument("--orgs", type=str, default=orgs)
parser.add_argument("--max_seq_len", type=int, default=1000, help="max len of protein sequence")
parser.add_argument("--min_seq_len", type=int, default=16, help="min len of protein sequence")
if orgs=='human':
    parser.add_argument("--max_domain_len", type=int, default=357, help="min len of human protein domain")
else:
    parser.add_argument("--max_domain_len", type=int, default=40, help="min len of yeast protein domain")

def get_config():
    args = parser.parse_args()
    return args
