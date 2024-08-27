# Training step information for DGQA
current_training_step = None

def set_training_step(step):
    global current_training_step
    current_training_step = step

def get_training_step():
    return current_training_step