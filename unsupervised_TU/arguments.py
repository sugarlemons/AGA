import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', default='BZR', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const',
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
            help='Learning rate.')
    parser.add_argument('--dnodes-degree', dest='dnodes_degree', type=float, default=0,
            help='degree of dnodes.')
    parser.add_argument('--pedges-degree', dest='pedges_degree', type=float, default=0,
            help='degree of pedges.')
    parser.add_argument('--subgraph-degree', dest='subgraph_degree', type=float, default=0,
            help='degree of subgraph.')
    parser.add_argument('--mask-nodes-degree', dest='mask_nodes_degree', type=float, default=0,
            help='degree of mask_nodes.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')

    parser.add_argument('--aug', dest='aug', type=str, default='dnodes')
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)

    return parser.parse_args()

