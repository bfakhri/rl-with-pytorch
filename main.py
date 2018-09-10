import argparse
import torch
import gym

# Our modules
from model import Model
from trainer import Trainer
from replay_buffer import ReplayBuffer

def main():
    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')
    # Training/Env Params
    parser.add_argument('--env-id', type=str, default='Pong-v0', metavar='ID',
                        help='Environment ID to train on')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--max-steps', type=int, default=100000, metavar='MS',
                        help='How many TOTAL environment steps to train for')
    parser.add_argument('--steps-to-optimize', type=int, default=100, metavar='OS',
                        help='How many environment steps to take before taking optimizer step')
    # Optimizer/Torch Params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # Loading/Saving Params
    parser.add_argument('--load-policy', type=bool, default=False, metavar='LP',
                        help='Boolean whether to load policy or not (start from scratch)')
    parser.add_argument('--load-policy-name', type=str, default='./policy.pt', metavar='LN',
                        help='String for policy name to load')
    parser.add_argument('--save-policy', type=bool, default=False, metavar='SP',
                        help='Boolean whether to save policy or not')
    parser.add_argument('--save-policy-name', type=str, default='./policy.pt', metavar='SN',
                        help='String for policy name to save')
    args = parser.parse_args()

    # Decide whether to use cuda or not
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # Set Torch seed
    torch.manual_seed(args.seed)
    # Set Torch device
    device = torch.device("cuda" if use_cuda else "cpu")

    # Init Environment
    train_env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)

    # Load/Init policy
    if(args.load_policy):
        # ---- Note yet implemented
        print('Loading not yet implemented')
        model = model.to(device)
    else:
        model = Model().to(device)

    # Init trainer
    trainer = Trainer(model=model, lr=args.lr, momentum=args.momentum)

    # Init replay buffer
    replay_buffer = ReplayBuffer(obs_shape=train_env.observation_space.shape, act_shape=train_env.action_space.n)

    # Train the model for max_steps number of steps
    for step in range(args.max_steps):
        # step the model/env
        # gather rollout

        if(step % args.steps_to_optimize == 0):
            # Run optimizer step
            trainer.train_step(model, replay_buffer)
        if(step % args.steps_per_optimize == 0):
            # Run evaluation for x episodes 
            for ep in range(args.eval_episodes):
                # step mode/eval env 
                # gather stats
                print('')

            # Set model to training mode once done evaluating
            model.train()



    # Save the model
    if(args.save_policy):
        # ---- Note yet implemented
        print('Saving not yet implemented')
    else:
        return 0




if __name__ == '__main__':
    main()
