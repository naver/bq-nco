"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

def add_common_args(parser):
    # -- Network --
    parser.add_argument("--dim_emb", type=int, default=192, help="Embeddings size")
    parser.add_argument("--dim_ff", type=int, default=512, help="FF size")
    parser.add_argument("--nb_layers_encoder", type=int, default=9, help="Encoder layers")
    parser.add_argument("--nb_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout")
    parser.add_argument("--use_batchnorm", dest='batchnorm', action='store_true',
                        help="True = BatchNorm, False = ReZero in encoder")
    parser.add_argument("--activation_ff", type=str, default="relu", help="ReLu or GeLu")
    parser.add_argument("--activation_attention", type=str, default="softmax", help="Softmax or 1.5-entmax")

    # -- Data --
    parser.add_argument("--train_dataset", type=str, help="Training dataset")
    parser.add_argument("--val_dataset", type=str, help="Validation dataset")
    parser.add_argument("--test_dataset", type=str, help="Test datsets")
    parser.add_argument("--output_dir", type=str, default="output/", help="Output dir")

    # -- Reload --
    parser.add_argument("--pretrained_model", type=str, default="", help="Load specific model for testing. ")
    # -- Eval --
    parser.add_argument("--beam_size", type=int, default=1, help="Number of beams, =1 for greedy search")
    parser.add_argument("--knns", type=int, default=-1, help="Number of KNNs used during the decoding")
    # -- Common --
    parser.add_argument("--test_only", dest='test_only', action='store_true')
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--train_batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=1024, help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=1024, help="Test batch size")
    parser.add_argument("--debug", dest='debug', action='store_true')

    parser.add_argument("--decay_rate", type=float, default=0.99, help="Decay rate for update lr")
    parser.add_argument("--decay_every", type=float, default=50, help="Each _ epochs, decay lr")


def add_common_training_args(parser):
    # -- Optim --
    parser.add_argument("--test_every", type=int, default=100, help="Test every n epochs")
    parser.add_argument("--nb_total_epochs", type=int, default=1000,
                        help="Number of epochs of trajectory learning (at each step, if 2-step learning)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
