import numpy as np
from collections import deque
import random

class PerceptionModule:
    def perceive(self, grid):
        return np.array(grid)

class AttentionModule:
    def select(self, perceptual_data):
        # Naive saliency selection: prioritize rewards
        saliency_map = np.where(perceptual_data == 2, 1, 0)
        return saliency_map

class WorkingMemory:
    def __init__(self, capacity=5):
        self.memory = deque(maxlen=capacity)
    
    def update(self, info):
        self.memory.append(info)

class LongTermMemory:
    def __init__(self):
        self.knowledge = {}

    def store(self, state, action):
        self.knowledge[state] = action

    def retrieve(self, state):
        return self.knowledge.get(state, None)

class DecisionModule:
    def decide(self, wm, ltm):
        last_state = wm.memory[-1] if wm.memory else None
        if last_state and ltm.retrieve(last_state):
            return ltm.retrieve(last_state)
        return random.choice(['up', 'down', 'left', 'right'])

class ActionModule:
    def act(self, action, position):
        x, y = position
        if action == 'up': return x-1, y
        if action == 'down': return x+1, y
        if action == 'left': return x, y-1
        if action == 'right': return x, y+1
        return position

class SimpleGridEnvironment:
    def __init__(self, size=10, num_threats=5, num_rewards=3):
        self.size = size
        self.grid = np.zeros((size, size)) # 0: empty, 1: threat, 2: reward
        self.agent_pos = [np.random.randint(0, size), np.random.randint(0, size)]
        self._place_objects(num_threats, 1) # Place threats
        self._place_objects(num_rewards, 2) # Place rewards
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 0 # Ensure agent starts on empty

    def _place_objects(self, num_objects, object_type):
        for _ in range(num_objects):
            while True:
r, c = np.random.randint(0, self.size), np.random.randint(0, self.size)
                if self.grid[r, c] == 0 and (r,c) != tuple(self.agent_pos):
                    self.grid[r, c] = object_type
                    break
    def get_perception(self, view_radius=1):
        r, c = self.agent_pos
        perception = {}
        for dr in range(-view_radius, view_radius + 1):
            for dc in range(-view_radius, view_radius + 1):
                if dr == 0 and dc == 0: continue 
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    perception[(dr, dc)] = self.grid[nr, nc]
                else:
                    perception[(dr, dc)] = -1 # Wall/boundary
        return perception

    def move_agent(self, dr, dc):
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return -10 
        self.agent_pos = [nr, nc]
        cell_value = self.grid[nr, nc]
        if cell_value == 1: 
            self.grid[nr, nc] = 0 
            return -50 
        elif cell_value == 2: 
            self.grid[nr, nc] = 0 
            return 100
        return -1

class AttentionModule:
    def __init__(self, threat_salience=2.0, reward_salience=1.5, distance_decay=0.5):
        self.threat_salience = threat_salience
        self.reward_salience = reward_salience
        self.distance_decay = distance_decay

    def attend(self, perception_data):
        attended_stimuli = {}
        if not perception_data:
            return attended_stimuli
        for loc, obj_type in perception_data.items():
            distance = np.sqrt(loc[0]**2 + loc[1]**2)
            if distance == 0: distance = 1
            base_salience = 0
            if obj_type == 1: # Threat
                base_salience = self.threat_salience
            elif obj_type == 2: # Reward
                base_salience = self.reward_salience
            attention_score = base_salience * (self.distance_decay ** (distance -1))
            if attention_score > 0.1 : 
                 attended_stimuli[loc] = attention_score
        return attended_stimuli
class WorkingMemoryModule:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.memory = [] 
        self.current_time = 0

    def update_time(self):
        self.current_time +=1

    def add_to_memory(self, item, item_type="generic"):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0) # FIFO
        self.memory.append({'data': item, 'type': item_type, 'time': self.current_time})

    def retrieve_relevant_memories(self, query_type=None, max_age=None):
        relevant = []
        for mem_item in self.memory:
            is_relevant = True
            if query_type and mem_item['type'] != query_type:
                is_relevant = False
            if max_age and (self.current_time - mem_item['time']) > max_age:
                is_relevant = False
            if is_relevant:
                relevant.append(mem_item)
        return relevant

    def remember_safe_spot(self, agent_pos):
        self.add_to_memory(tuple(agent_pos), item_type="safe_spot")
class DecisionMakingModule: # V2
    def __init__(self, exploration_rate=0.1):
        self.exploration_rate = exploration_rate

    def decide_action_V2(self, current_perception, attended_stimuli, memories, agent_pos_tuple, possible_moves):
        action_scores = {move: 0.0 for move in possible_moves}
        for move in possible_moves:
            target_loc_relative = move 
            if target_loc_relative in attended_stimuli:
                obj_type = current_perception.get(target_loc_relative) 
                attention_score = attended_stimuli[target_loc_relative]
                if obj_type == 1: 
                    action_scores[move] -= attention_score * 10 
                elif obj_type == 2: 
                    action_scores[move] += attention_score * 5  
            if target_loc_relative not in current_perception or current_perception.get(target_loc_relative) == -1:
                action_scores[move] -= 2 
        recent_safe_spots = [mem['data'] for mem in memories if mem['type'] == 'safe_spot']
        for move in possible_moves:
            target_abs_pos = (agent_pos_tuple[0] + move[0], agent_pos_tuple[1] + move[1])
            if target_abs_pos in recent_safe_spots:
                current_object_at_target = current_perception.get(move)
                if not (current_object_at_target == 1 and move in attended_stimuli and attended_stimuli[move] > 0.5):
                     action_scores[move] += 2 
        if random.random() < self.exploration_rate or not any(v != 0 for v in action_scores.values()):
            valid_moves = [m for m in possible_moves if current_perception.get(m, -1) != -1]
            if not valid_moves: return random.choice(possible_moves) 
            return random.choice(valid_moves)
        else:
            best_score = -float('inf')
            best_moves = []
            non_wall_moves = {m:s for m,s in action_scores.items() if current_perception.get(m,-1) != -1}
            target_scores = non_wall_moves if non_wall_moves else action_scores
            for move, score in target_scores.items():
                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)
            if not best_moves: 
                 valid_moves = [m for m in possible_moves if current_perception.get(m, -1) != -1]
                 if not valid_moves: return random.choice(possible_moves)
                 return random.choice(valid_moves)
            return random.choice(best_moves)
class CognitiveAgent:
    def __init__(self, environment):
        self.env = environment
        self.attention = AttentionModule()
        self.wm = WorkingMemoryModule(capacity=10)
        self.decision_maker = DecisionMakingModule(exploration_rate=0.1)
        self.score = 0
        self.possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                               (1,1), (1,-1), (-1,1), (-1,-1)] 

    def run_step(self):
        self.wm.update_time() 
        current_perception = self.env.get_perception(view_radius=2)
        attended_features = self.attention.attend(current_perception)
        is_currently_safe = True
        if attended_features: 
            for loc_key, score in attended_features.items():
                obj_type = current_perception.get(loc_key) 
                if obj_type == 1 and score > self.attention.threat_salience * 0.5 : 
                    is_currently_safe = False
                    break
        if is_currently_safe:
            self.wm.remember_safe_spot(tuple(self.env.agent_pos))
        recent_memories = self.wm.retrieve_relevant_memories(max_age=20) 
        chosen_move = self.decision_maker.decide_action_V2(
            current_perception, attended_features, recent_memories, 
            tuple(self.env.agent_pos), self.possible_moves
        )
        reward_penalty = self.env.move_agent(chosen_move[0], chosen_move[1])
        self.score += reward_penalty

# Main execution (Example)
if __name__ == "__main__":
    grid_env = SimpleGridEnvironment(size=10, num_threats=7, num_rewards=4)
    agent = CognitiveAgent(grid_env)
    print(f"Initial Grid: (Agent at {agent.env.agent_pos})")
    print(agent.env.grid)
    print("-" * 20)
    num_steps = 50
    for i in range(num_steps):
        old_pos = agent.env.agent_pos[:] 
        agent.run_step()
        if (i+1) % 10 == 0 or i == num_steps -1 : 
            print(f"\n--- State after Step {i+1} ---")
            print(f"Agent at {agent.env.agent_pos}. Current Score: {agent.score}")
            display_grid = agent.env.grid.astype(str) 
            display_grid[display_grid == '0.0'] = '.'
            display_grid[display_grid == '1.0'] = 'T' 
            display_grid[display_grid == '2.0'] = 'R' 
            if 0 <= agent.env.agent_pos[0] < agent.env.size and \
               0 <= agent.env.agent_pos[1] < agent.env.size:
                display_grid[agent.env.agent_pos[0], agent.env.agent_pos[1]] = 'A' 
            print("Grid View:")
            for row in display_grid: print(' '.join(row))
            print("-" * 20)
    print(f"\n=== Run Complete ===")
    print(f"Final Score after {num_steps} steps: {agent.score}")
    print("Final Agent Position:", agent.env.agent_pos)
    print("Final Grid State:")
    final_display_grid = agent.env.grid.astype(str)
    final_display_grid[final_display_grid == '0.0'] = '.'
    final_display_grid[final_display_grid == '1.0'] = 'T'
    final_display_grid[final_display_grid == '2.0'] = 'R'
    if 0 <= agent.env.agent_pos[0] < agent.env.size and \
        0 <= agent.env.agent_pos[1] < agent.env.size:
        final_display_grid[agent.env.agent_pos[0], agent.env.agent_pos[1]] = 'A'
    for row in final_display_grid: print(' '.join(row))
