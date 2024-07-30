import matplotlib.pyplot as plt
import torch as th
import wandb
from mu_alpha_zero import MuZeroNet
from mu_alpha_zero.AlphaZero.Arena.arena import Arena
from mu_alpha_zero.AlphaZero.Arena.players import NetPlayer, HumanPlayer
from mu_alpha_zero.Hooks.hook_callables import HookMethodCallable
from mu_alpha_zero.Hooks.hook_manager import HookManager
from mu_alpha_zero.Hooks.hook_point import HookPoint, HookAt
from mu_alpha_zero.MuZero.muzero import MuZero
from mu_alpha_zero.MuZero.utils import mz_optuna_parameter_search
from mu_alpha_zero.config import MuZeroConfig
from mu_alpha_zero.mem_buffer import MemBuffer
from mu_alpha_zero.MuZero.utils import muzero_loss

from game_env import GameEnv



def train_mz():
    game = GameEnv(6, 7, 4, True)
    config = MuZeroConfig()
    hook_manager = HookManager()
    hook_cls = CollectInfo()
    hook_manager.register_method_hook(HookPoint(HookAt.MIDDLE, "networks.py", "train_net"),
                                      HookMethodCallable(hook_cls.training_hook, ()))
    wandb.login(key="***REMOVED***")
    config.num_steps = 100_000
    config.self_play_games = 3
    config.epochs = 1500
    config.num_simulations = 50
    config.num_iters = 40
    config.num_pit_games = 20
    config.num_workers = 1
    config.K = 5
    config.alpha = 1
    config.beta = 1
    config.max_buffer_size = 500_000
    config.update_threshold = 0.6
    config.c = 1.25
    config.use_pooling = False
    config.tau = 1
    config.gamma = 1
    config.frame_skip = 1
    config.frame_buffer_size = 8
    config.zero_tau_after = 100
    config.net_latent_size = [6, 7]
    config.enable_per = False
    config.az_net_linear_input_size = [10752, 10752, 10752]
    config.resize_images = False
    config.scale_state = False
    config.multiple_players = True
    config.arena_running_muzero = True
    config.rep_input_channels = 24
    config.unravel = False
    config.requires_player_to_reset = True
    config.use_original = True
    config.num_blocks = 8
    config.lr = 8e-4
    config.log_dir = "/home/skyr/Downloads/connect4_logs"
    config.checkpoint_dir = "/home/skyr/Downloads/connect4_nets"
    config.batch_size = 256
    config.support_size = 1
    config.l2 = 1e-3
    config.state_linear_layers = 1
    config.v_linear_layers = 1
    config.pi_linear_layers = 1
    config.net_dropout = 0.2
    config.dirichlet_alpha = 0
    config.enable_frame_buffer = True
    config.frame_buffer_ignores_actions = True
    config.actions_are = "columns"
    config.num_td_steps = 42
    # config._value_reward_loss = lambda y_hat, y: th.sum(th.nn.functional.mse_loss(y_hat, y, reduction="none"), dim=1).unsqueeze(1)
    config._value_reward_loss = muzero_loss
    config.loss_gets_support = True
    # config.lr_scheduler = th.optim.lr_scheduler.CyclicLR
    # config.lr_scheduler_kwargs = {"max_lr": 1e-3,"base_lr": 4e-4, "step_size_up": 1000, "step_size_down": 1000,"cycle_momentum": False}
    wandb.init(project=config.wandbd_project_name, config=config.to_dict())
    mz = MuZero(game)
    memory = MemBuffer(config.max_buffer_size, disk=False, full_disk=False, dir_path=None)
    arena = Arena(game, config, th.device("cuda" if th.cuda.is_available() else "cpu"), hook_manager=hook_manager,
                  state_managed=True)
    mz.create_new(config, MuZeroNet, memory, arena_override=arena, checkpointer_verbose=True)
    mz.train()


def find_hyperparams():
    game = GameEnv(6, 7, 4, True)
    config = MuZeroConfig()
    wandb.login(key="***REMOVED***")
    config.num_steps = 100_000
    config.self_play_games = 3000
    config.epochs = 5500
    config.num_simulations = 400
    config.num_iters = 40
    config.num_pit_games = 40
    config.num_workers = 10
    config.K = 5
    config.alpha = 1
    config.beta = 1
    config.max_buffer_size = 500_000
    config.update_threshold = 0.6
    config.c = 1.25
    config.use_pooling = False
    config.tau = 1
    config.gamma = 1
    config.frame_skip = 1
    config.frame_buffer_size = 32
    config.zero_tau_after = 100
    config.net_latent_size = [4, 4]
    config.az_net_linear_input_size = [4, 8, 16]
    config.resize_images = False
    config.scale_state = False
    config.multiple_players = True
    config.arena_running_muzero = True
    config.rep_input_channels = 96
    config.unravel = False
    config.requires_player_to_reset = True
    config.use_original = True
    config.num_blocks = 5
    config.lr = 1e-4
    config.batch_size = 512
    config.l2 = 1e-5
    config.state_linear_layers = 1
    config.v_linear_layers = 1
    config.num_td_steps = 42
    config.pi_linear_layers = 1
    config.net_dropout = 0.2
    config.net_action_size = int(game.get_num_actions())
    wandb.init(project="MuZero 2", config=config.to_dict())
    memory = MemBuffer(config.max_buffer_size, disk=False, full_disk=False, dir_path=None)
    arena = Arena(game, config, th.device("cuda" if th.cuda.is_available() else "cpu"), hook_manager=None,
                  state_managed=True)
    mz_optuna_parameter_search(100, None, "mz_connect4_param_search", game, config, in_memory=True,
                               arena_override=arena, memory_override=memory)


def play_mz():
    game = GameEnv(6, 7, 4, False)
    mz = MuZero(game)
    mz.from_checkpoint(MuZeroNet, MemBuffer(10_000, dir_path=None),
                       "/home/skyr/Downloads/improved_net_19.pth",
                       "Checkpoints", False)
    # mz.trainer.muzero_alphazero_config.num_simulations = 800
    mz.trainer.muzero_alphazero_config.arena_tau = 1
    arena = Arena(game, mz.trainer.muzero_alphazero_config, th.device("cuda" if th.cuda.is_available() else "cpu"),
                  state_managed=True)
    p1 = NetPlayer(game, **{"network": mz.trainer.network, "monte_carlo_tree_search": mz.trainer.mcts})
    p2 = HumanPlayer(game)
    p1.network.eval()
    arena.pit(p1, p2, 10, 800, debug=True)


class CollectInfo:
    def __init__(self):
        self.losses = []
        self.error_experiences = []
        self.num = 0

    def training_hook(self, cls: object, *args):
        self.losses.append(args[1])
        if args[-1] % 500 != 0:
            return
        plt.plot(self.losses)
        plt.savefig(f"/workspace/6TakesGameEnv/Logs/losses{self.num}.png")
        exp_batch = args[0]
        loss_r = args[4]
        max_error = th.argmax(loss_r).item()
        self.error_experiences.append(exp_batch[max_error])
        exp_batch[max_error][-1].make_persistent()
        self.num += 1

    def save_plot(self):
        plt.plot(self.losses)
        plt.savefig("/auto/vestec1-elixir/home/vvlcek/Logs/losses.png")

    def save_error_experiences(self):
        with open("/auto/vestec1-elixir/home/vvlcek/Logs/error_experiences.pkl", "wb") as file:
            th.save(self.error_experiences, file)

    def plot_random_img(self):
        data = th.load("/auto/vestec1-elixir/home/vvlcek/Logs/error_experiences.pkl")
        random_idx = th.randint(0, len(data), (1,)).item()
        img = data[random_idx][-1]
        plt.imshow(img)


if __name__ == "__main__":
    train_mz()
    # play_mz()
    # find_hyperparams()
