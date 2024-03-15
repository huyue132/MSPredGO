from config import get_config

args = get_config()
if args.orgs=='human':
    OUT_nodes = {'BP': 491,'MF': 321,'CC': 240}
else:
    OUT_nodes = {'BP': 491,'MF': 321,'CC': 240}

CFG = {
    'cfg00': [16, 'M', 16, 'M'],
    'cfg01': [16, 'M', 32, 'M'],
    'cfg02': [32, 'M'],
    'cfg03': [64, 'M'],
    'cfg04': [16, 'M', 16, 'M',32, 'M'],
    'cfg05': [64, 'M', 32, 'M',16, 'M'],
    'cfg06': [64, 'M', 32, 'M',32, 'M'],
    'cfg07': [128, 'M', 64, 'M2'],
    'cfg08': [512, 'M', 128, 'M2',32, 'M2'],
    'cfg09': [128, 'M', 64, 'M'],
    'cfg10': [64, 'M'],
    'cfg11': [32, 'M'],
    'cfg12': [512, 'M2', 128, 'M2', 32, 'M']
}
Thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
              0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
              0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
              0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
              0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
              0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
              0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
              0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
              0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
              0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

def get_annotation_vec(func, anno_file):
    annotations = {}
    with open(anno_file, 'r') as f:
        num = 1
        for line in f:
            if num == 1:
                num = 2
                continue
            items = line.strip().split(',')
            if args.orgs == 'human':
                if func == 'BP':
                    prot, annotation = items[0], items[1:492]
                elif func == 'MF':
                    prot, annotation = items[0], items[492:813]
                elif func == 'CC':
                    prot, annotation = items[0], items[813:]
                annotations[prot] = annotation
            else:
                if func == 'BP':
                    prot, annotation = items[0], items[1:374]
                elif func == 'MF':
                    prot, annotation = items[0], items[374:545]
                elif func == 'CC':
                    prot, annotation = items[0], items[545:]
                annotations[prot] = annotation
    return annotations
