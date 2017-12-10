def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--replay_size', type=int, default=100000, help='replay memory size')
    parser.add_argument('--update_time', type=int, default=3000, help='update time for training')
    parser.add_argument('--skip', type=int, default=3, help='timestep to skip for training')
    parser.add_argument('--gamma', type=float, default=.99, help='gamma value for DQN')
    
    parser.add_argument('--discount_factor', type=float, default=.99, help='discount factor for PG')
    
    
    return parser
