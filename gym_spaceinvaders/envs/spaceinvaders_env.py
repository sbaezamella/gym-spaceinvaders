import glob
import os

import atari_py
import cv2 as cv
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


class CustomSpaceInvadersEnv(gym.Env, utils.EzPickle):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        game="space_invaders",
        mode=None,
        difficulty=None,
        obs_type="distance",
        frameskip=(2, 5),
        repeat_action_probability=0.25,
        full_action_space=False,
    ):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(
            self, game, mode, difficulty, obs_type, frameskip, repeat_action_probability
        )
        assert obs_type in ("distance")

        self.game = game
        self.game_path = atari_py.get_game_path(game)
        self.game_mode = mode
        self.game_difficulty = difficulty

        if not os.path.exists(self.game_path):
            msg = "You asked for game %s but path %s does not exist"
            raise IOError(msg % (game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.ale = atari_py.ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(
            repeat_action_probability, (float, int)
        ), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat(
            "repeat_action_probability".encode("utf-8"), repeat_action_probability
        )

        self.seed()

        self._action_set = (
            self.ale.getLegalActionSet()
            if full_action_space
            else self.ale.getMinimalActionSet()
        )
        self.action_space = spaces.Discrete(len(self._action_set))

        self._obs_type == "distance"
        self.observation_space = spaces.Box(
            low=0, high=200, shape=(10,), dtype=np.float32
        )
        self.images_folder = os.path.join(os.path.dirname(__file__), "images")
        self.gray_enemies_dict_from_images(self.images_folder)
        self.gray_agent_array_from_image(self.images_folder)
        self.locs = {}

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b"random_seed", seed2)
        self.ale.loadROM(self.game_path)

        if self.game_mode is not None:
            modes = self.ale.getAvailableModes()

            assert self.game_mode in modes, (
                'Invalid game mode "{}" for game {}.\nAvailable modes are: {}'
            ).format(self.game_mode, self.game, modes)
            self.ale.setMode(self.game_mode)

        if self.game_difficulty is not None:
            difficulties = self.ale.getAvailableDifficulties()

            assert self.game_difficulty in difficulties, (
                'Invalid game difficulty "{}" for game {}.\nAvailable difficulties are: {}'
            ).format(self.game_difficulty, self.game, difficulties)
            self.ale.setDifficulty(self.game_difficulty)

        return [seed1, seed2]

    def step(self, a):
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.ale.act(action)
        ob = self._get_obs()

        return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def _get_image(self):
        return self.ale.getScreenRGB2()

    def _get_distances(self):
        # Obtener tablero en gris
        gray_board = self.get_gray_board()
        # Buscar coordenadas de enemigos
        enemy_coords = self.get_enemy_coordinates(gray_board)
        # Retornar todas las coordenadas
        return enemy_coords

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        return self._get_distances()

    def get_gray_board(self):
        # Return gray image and reshape ndarray
        return self.ale.getScreenGrayscale().reshape(210, 160)

    def get_enemy_coordinates(self, board):
        enemy_coords = []
        agent_centroid, found_agent = self.find_agent_in_board(board)
        if found_agent:
            enemies_centroids = self.find_enemies_in_board(board)
            # print(f"Agent centroid")
            # print(agent_centroid)
            # print(f"Enemies centroids")
            # print(enemies_centroids)
            # agent_x, agent_y = agent_centroid
            # coords = enemies_centroids.copy()
            # coords[:, 0], coords[:, 1] = coords[:, 0] - agent_x, coords[:, 1] - agent_y
            # coords = np.abs(coords)

            # distances = np.square(coords)
            # distances = np.sqrt(np.sum(distances, axis=1, keepdims=True))
            # coords = np.append(coords, distances, axis=1)

            # sorted_enemy_coords = coords[np.argsort(coords[:, 2])]

            # nearest_enemies = sorted_enemy_coords[:5, :2].reshape(
            #     10,
            # )

            # return nearest_enemies

    def gray_enemies_dict_from_images(self, images_folder):
        self.enemies_dict = {
            f"enemy_{i}": [
                cv.imread(f"{images_folder}/enemy_{i}.png", 0),
                cv.imread(f"{images_folder}/enemy_{i}{i}.png", 0),
            ]
            for i in range(6)
        }
        # self.enemies_dict = {
        #     image.split("/")[-1].split(".")[0]: cv.imread(image, 0)
        #     for image in glob.glob(f"{images_folder}/enemy*.png")
        # }

    def gray_agent_array_from_image(self, images_folder):
        self.agent_array = cv.imread(f"{images_folder}/agent.png", 0)

    def find_agent_in_board(self, board):
        agent = self.agent_array
        w, h = agent.shape[::-1]
        loc = self._apply_match_template(board, agent)
        if loc.any():
            y, x = loc[0]
            centroid = ((2 * x + w) / 2, (2 * y + h) / 2)
            return centroid, True
        return None, False

    def find_enemies_in_board(self, board):
        enemies = self.enemies_dict
        centroids = []
        for key, enemy in enemies.items():
            enemyv1, enemyv2 = enemy
            loc1 = self._apply_match_template(board, enemyv1)
            loc2 = self._apply_match_template(board, enemyv2)
            if loc1.any():
                loc = loc1
                w, h = enemyv1.shape[::-1]
            elif loc2.any():
                loc = loc2
                w, h = enemyv2.shape[::-1]
            print("loc")
            print(loc)
            print()

            if isinstance(self.locs.get(key), np.ndarray):
                old = self.locs[key]
                if not np.array_equal(old, loc):
                    if loc.shape == old.shape:
                        self.locs[key] = loc
                    elif loc.shape[0] < old.shape[0]:
                        print(loc.shape)
                        print(old.shape)

                        destroyed = np.isin(old, [250, 250]).sum() // 2
                        masks = [
                            np.isin(old[:, 1], loc),
                            np.isin(old[:, 1] + 1, loc),
                            np.isin(old[:, 1] - 1, loc),
                        ]
                        # print("key")
                        # print(key)
                        # print()
                        # print("old")
                        # print(old)
                        # print()
                        # print("loc")
                        # print(loc)
                        # print()
                        # print("old")
                        # print(np.isin(old[:, 1], loc))
                        # print()
                        # print("old + 1")
                        # print(np.isin(old[:, 1] + 1, loc))
                        # print()
                        # print("old -1")
                        # print(np.isin(old[:, 1] - 1, loc))
                        # print()
                        for mask in masks:
                            if self.ale.getFrameNumber() > 125:
                                pass
                                # print("mask")
                                # print(mask)
                                # print()
                            if mask.any():
                                missing = np.size(mask) - np.count_nonzero(mask)
                                if missing != destroyed:

                                    idxs = np.where(mask == 0)[0]
                                    # print(f"{idxs=}")
                                    for idx in idxs:
                                        loc = np.insert(loc, idx, [250, 250], axis=0)
                        self.locs[key] = loc

            else:
                self.locs[key] = loc

            # if loc.any():
            #     if hasattr(self.locs.get(key), "shape"):
            #         old = self.locs.get(key)
            #         if loc.shape != old.shape:
            #             # destroyed = np.isin(old, [-1, -1]).sum()
            #             # if destroyed:

            #             mask = np.isin(old[:, 1], loc)
            #             print(f"{key=}")
            #             print(f"{old=}")
            #             print(f"{loc=}")
            #             # print(f"{mask=}")
            #             if (np.size(mask) - np.count_nonzero(mask)) == 1:
            #                 idx = np.where(mask == 0)[0]
            #                 print(f"{idx=}")
            #                 loc = np.insert(loc, idx, [-1, -1], axis=0)
            #             else:
            #                 pass

            #         self.locs[key] = loc
            #     else:
            #         self.locs[key] = loc
            #     # print(loc)
            #     # print()
            #     loc = loc * 2
            #     loc[:, 0] = loc[:, 0] + h
            #     loc[:, 1] = loc[:, 1] + w
            #     loc = loc / 2
            #     centroids.append(loc)
        # print(self.locs)
        # TODO: Check for destroyed enemy spaceship and fill void with -1
        # centroids = np.array(centroids).reshape(36, 2)
        # # Swap columns
        # centroids[:, 0], centroids[:, 1] = centroids[:, 1], centroids[:, 0].copy()
        # return centroids

    def _apply_match_template(self, img, template, threshold=0.8):
        # Apply template Matching
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)

        loc = np.argwhere(res >= threshold)
        return loc

    def reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def render(self, mode="human"):
        img = self._get_image()
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            "UP": ord("w"),
            "DOWN": ord("s"),
            "LEFT": ord("a"),
            "RIGHT": ord("d"),
            "FIRE": ord(" "),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
