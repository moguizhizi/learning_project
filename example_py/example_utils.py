class RewardModel:
    def compute_reward(self, text):
        return len(text) / 10.0

class RLHFWorkerExtension:
    def __init__(self):
        self.reward_model = RewardModel()

    def pre_process(self, input_prompt):
        print(f"Worker: Pre-processing input: {input_prompt}")
        return f"[RLHF] {input_prompt}"

    def post_process(self, output_text):
        reward = self.reward_model.compute_reward(output_text)
        print(f"Worker: Post-processing output: {output_text}, Reward: {reward}")
        return {"text": output_text, "reward": reward}