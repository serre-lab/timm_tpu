from hgru import hConvGruCell, hConvGru, hConvGruResNet

def create_hgru(time_steps = 8):
    model = hConvGru(time_steps)
    return model

